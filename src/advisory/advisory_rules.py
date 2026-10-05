"""
advisory_rules.py
=================
GKMS (Gram-level Krishi Mausam Sewa) Advisory Rule Engine.

Translates downscaled panchayat-level weather predictions into actionable,
ICAR-aligned agronomic advisories. All alerts are bilingual (English / Hindi).

Rule Design Principles
----------------------
* Rules are **priority-ordered**: higher-severity alerts take precedence when
  multiple conditions fire simultaneously.
* Each rule is a pure function: ``condition(weather) → bool``. This makes the
  engine trivially testable and extendable without modifying existing rules.
* Bilingual strings follow the ICAR Agromet Advisory Service (AAS) vocabulary.

Weather Variable Schema (input dict)
--------------------------------------
    tmax     : float  °C   — maximum 2-m air temperature
    tmin     : float  °C   — minimum 2-m air temperature
    rainfall : float  mm   — daily accumulated precipitation
    rh       : float  %    — relative humidity (%)
    wind     : float  m/s  — surface wind speed

Usage
-----
    from src.advisory.advisory_rules import evaluate_advisory

    weather = {"tmax": 36.5, "tmin": 24.0, "rainfall": 28.0, "rh": 87.0, "wind": 3.2}
    alerts  = evaluate_advisory(weather, panchayat_id="RJ_CH_001")

    for alert in alerts:
        print(alert["severity"], "|", alert["message_en"])
        print("  हिन्दी:", alert["message_hi"])
"""

from __future__ import annotations

import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdvisoryAlert:
    """
    A single agro-meteorological advisory alert.

    Attributes
    ----------
    alert_id     : str   Unique rule identifier (stable across API versions).
    category     : str   Alert category (Temperature / Rainfall / Humidity /
                         Wind / Pest-Disease / Irrigation / Composite).
    severity     : str   'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
    message_en   : str   Advisory message in English.
    message_hi   : str   Advisory message in Hindi (Devanagari).
    action_en    : str   Recommended farmer action (English).
    action_hi    : str   Recommended farmer action (Hindi).
    triggered_by : dict  Weather variable(s) that triggered the rule.
    """
    alert_id    : str
    category    : str
    severity    : str
    message_en  : str
    message_hi  : str
    action_en   : str
    action_hi   : str
    triggered_by: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id":     self.alert_id,
            "category":     self.category,
            "severity":     self.severity,
            "message_en":   self.message_en,
            "message_hi":   self.message_hi,
            "action_en":    self.action_en,
            "action_hi":    self.action_hi,
            "triggered_by": self.triggered_by,
        }


# Severity ordering for sorting
_SEVERITY_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


# ---------------------------------------------------------------------------
# Individual rule functions
# Returns AdvisoryAlert | None
# ---------------------------------------------------------------------------

def _rule_heavy_rainfall(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Rainfall > 50 mm — heavy rain / flash-flood risk."""
    if w["rainfall"] > 50.0:
        return AdvisoryAlert(
            alert_id="RAIN_HEAVY",
            category="Rainfall",
            severity="CRITICAL",
            message_en=(
                "Heavy rainfall expected (>{:.0f} mm). "
                "Risk of waterlogging and field erosion.".format(w["rainfall"])
            ),
            message_hi=(
                "भारी वर्षा की संभावना (>{:.0f} मिमी)। "
                "जलभराव एवं खेत कटाव का खतरा।".format(w["rainfall"])
            ),
            action_en=(
                "Avoid all field operations. Ensure proper drainage in low-lying fields. "
                "Do not apply fertilizers or pesticides."
            ),
            action_hi=(
                "सभी खेत कार्य स्थगित करें। निचले खेतों में जल निकास सुनिश्चित करें। "
                "उर्वरक या कीटनाशक का प्रयोग न करें।"
            ),
            triggered_by={"rainfall_mm": w["rainfall"]},
        )
    return None


def _rule_moderate_rainfall_spray(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Rainfall > 20 mm — postpone spraying operations."""
    if 20.0 < w["rainfall"] <= 50.0:
        return AdvisoryAlert(
            alert_id="RAIN_SPRAY_POSTPONE",
            category="Rainfall",
            severity="HIGH",
            message_en=(
                "Moderate rainfall ({:.1f} mm). "
                "Spraying operations will be washed off.".format(w["rainfall"])
            ),
            message_hi=(
                "मध्यम वर्षा ({:.1f} मिमी)। "
                "छिड़काव कार्य धुल जाएगा।".format(w["rainfall"])
            ),
            action_en=(
                "Postpone all pesticide and foliar fertilizer spraying. "
                "Schedule spraying for dry days (rainfall < 5 mm)."
            ),
            action_hi=(
                "सभी कीटनाशक एवं पर्णीय उर्वरक छिड़काव स्थगित करें। "
                "शुष्क दिनों (वर्षा < 5 मिमी) में छिड़काव करें।"
            ),
            triggered_by={"rainfall_mm": w["rainfall"]},
        )
    return None


def _rule_light_rainfall_sow(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Rainfall 5–20 mm — favourable for sowing / transplanting."""
    if 5.0 <= w["rainfall"] <= 20.0:
        return AdvisoryAlert(
            alert_id="RAIN_SOW_FAVOURABLE",
            category="Rainfall",
            severity="LOW",
            message_en=(
                "Light to moderate rainfall ({:.1f} mm). "
                "Favourable soil moisture for sowing.".format(w["rainfall"])
            ),
            message_hi=(
                "हल्की से मध्यम वर्षा ({:.1f} मिमी)। "
                "बुआई के लिए अनुकूल नमी।".format(w["rainfall"])
            ),
            action_en=(
                "Good window for sowing Kharif crops (pearl millet, clusterbean, moth bean). "
                "Carry out gap-filling for standing crops."
            ),
            action_hi=(
                "खरीफ फसलों (बाजरा, ग्वार, मोठ) की बुआई के लिए उत्तम समय। "
                "खड़ी फसलों में रिक्त स्थान भरें।"
            ),
            triggered_by={"rainfall_mm": w["rainfall"]},
        )
    return None


def _rule_high_rh_fungal(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """RH > 80% + Tmax 25–35°C → fungal disease risk (rust, blast, blight)."""
    if w["rh"] > 80.0 and 25.0 <= w["tmax"] <= 38.0:
        return AdvisoryAlert(
            alert_id="DISEASE_FUNGAL_RUST",
            category="Pest-Disease",
            severity="HIGH",
            message_en=(
                "High relative humidity ({:.0f}%) combined with warm temperatures "
                "({:.1f}°C) creates conditions highly favourable for fungal diseases "
                "(rust, powdery mildew, blast).".format(w["rh"], w["tmax"])
            ),
            message_hi=(
                "उच्च आर्द्रता ({:.0f}%) एवं गर्म तापमान ({:.1f}°C) "
                "फफूंद रोगों (रतुआ, चूर्णिल आसिता, ब्लास्ट) के लिए "
                "अनुकूल परिस्थितियाँ।".format(w["rh"], w["tmax"])
            ),
            action_en=(
                "Spray Propiconazole 25% EC @ 0.1% or Mancozeb 75% WP @ 0.2% "
                "as a preventive measure. Scout fields for early symptoms. "
                "Avoid overhead irrigation."
            ),
            action_hi=(
                "प्रोपिकोनाजोल 25% ईसी @ 0.1% या मैन्कोजेब 75% डब्ल्यूपी @ 0.2% "
                "का निवारक छिड़काव करें। खेतों में प्रारंभिक लक्षणों की जांच करें। "
                "ऊपरी सिंचाई से बचें।"
            ),
            triggered_by={"rh_pct": w["rh"], "tmax_c": w["tmax"]},
        )
    return None


def _rule_very_high_rh_avoid_ops(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """RH > 90% → avoid field operations, poor spray efficacy."""
    if w["rh"] > 90.0:
        return AdvisoryAlert(
            alert_id="RH_VERY_HIGH",
            category="Humidity",
            severity="MODERATE",
            message_en=(
                "Very high relative humidity ({:.0f}%). "
                "Spray droplets will not dry; efficacy will be severely reduced.".format(w["rh"])
            ),
            message_hi=(
                "अत्यधिक सापेक्षिक आर्द्रता ({:.0f}%)। "
                "छिड़काव बूंदें सूखेंगी नहीं; प्रभावकारिता बेहद कम होगी।".format(w["rh"])
            ),
            action_en=(
                "Reschedule spraying and dusting operations. "
                "Early morning operations (before 7 AM) are preferable when humidity drops."
            ),
            action_hi=(
                "छिड़काव एवं धूलन कार्य पुनर्निधारित करें। "
                "प्रातःकाल (7 बजे से पहले) जब आर्द्रता कम हो, तब कार्य करना उचित है।"
            ),
            triggered_by={"rh_pct": w["rh"]},
        )
    return None


def _rule_extreme_heat(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Tmax > 42°C — extreme heat stress for crops and livestock."""
    if w["tmax"] > 42.0:
        return AdvisoryAlert(
            alert_id="TEMP_EXTREME_HEAT",
            category="Temperature",
            severity="CRITICAL",
            message_en=(
                "Extreme heat stress: Tmax = {:.1f}°C. "
                "Severe risk of crop wilting, pollen sterility, and livestock heat stroke.".format(
                    w["tmax"])
            ),
            message_hi=(
                "अत्यधिक ताप तनाव: अधिकतम तापमान = {:.1f}°C। "
                "फसल मुरझाने, पराग बाँझपन एवं पशु लू लगने का गंभीर खतरा।".format(w["tmax"])
            ),
            action_en=(
                "Provide shade nets (50% shade) over vegetable crops. "
                "Irrigate during evening hours to reduce soil temperature. "
                "Ensure adequate drinking water and shade for livestock."
            ),
            action_hi=(
                "सब्जी फसलों पर शेड नेट (50% छाया) लगाएं। "
                "मृदा तापमान कम करने के लिए शाम को सिंचाई करें। "
                "पशुओं के लिए पर्याप्त पेयजल एवं छाया सुनिश्चित करें।"
            ),
            triggered_by={"tmax_c": w["tmax"]},
        )
    return None


def _rule_moderate_heat(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Tmax 38–42°C — moderate heat advisory."""
    if 38.0 < w["tmax"] <= 42.0:
        return AdvisoryAlert(
            alert_id="TEMP_MODERATE_HEAT",
            category="Temperature",
            severity="MODERATE",
            message_en=(
                "High temperatures expected: Tmax = {:.1f}°C. "
                "Heat stress risk for flowering crops.".format(w["tmax"])
            ),
            message_hi=(
                "उच्च तापमान की संभावना: अधिकतम तापमान = {:.1f}°C। "
                "फूल वाली फसलों में ताप तनाव का खतरा।".format(w["tmax"])
            ),
            action_en=(
                "Apply light irrigation to reduce canopy temperature. "
                "Avoid urea application during peak heat hours (11 AM–3 PM)."
            ),
            action_hi=(
                "वितान तापमान कम करने के लिए हल्की सिंचाई करें। "
                "अत्यधिक गर्मी के समय (11 AM–3 PM) यूरिया का प्रयोग न करें।"
            ),
            triggered_by={"tmax_c": w["tmax"]},
        )
    return None


def _rule_cold_wave(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Tmin < 5°C — cold wave / frost risk for winter vegetables and nurseries."""
    if w["tmin"] < 5.0:
        return AdvisoryAlert(
            alert_id="TEMP_COLD_WAVE",
            category="Temperature",
            severity="HIGH",
            message_en=(
                "Cold wave conditions: Tmin = {:.1f}°C. "
                "Risk of frost damage to vegetables and nursery crops.".format(w["tmin"])
            ),
            message_hi=(
                "शीत लहर की स्थिति: न्यूनतम तापमान = {:.1f}°C। "
                "सब्जियों एवं नर्सरी फसलों को पाले से नुकसान का खतरा।".format(w["tmin"])
            ),
            action_en=(
                "Cover nursery beds and vegetable plots with polythene mulch or crop covers. "
                "Light irrigation before evening creates a latent heat barrier against frost. "
                "Light smoke screens in orchards can help."
            ),
            action_hi=(
                "नर्सरी बेड एवं सब्जी भूखंडों को पॉलीथीन मल्च या फसल आवरण से ढकें। "
                "शाम से पहले हल्की सिंचाई पाले के विरुद्ध ऊष्मा अवरोध बनाती है। "
                "बागों में हल्का धुआं उपयोगी हो सकता है।"
            ),
            triggered_by={"tmin_c": w["tmin"]},
        )
    return None


def _rule_high_wind_dust(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Wind > 12 m/s — strong wind / Andhi (dust storm) risk in Rajasthan."""
    if w["wind"] > 12.0:
        return AdvisoryAlert(
            alert_id="WIND_HIGH_ANDHI",
            category="Wind",
            severity="HIGH",
            message_en=(
                "Strong winds forecast: {:.1f} m/s. "
                "Risk of Andhi (dust storm) affecting crop health.".format(w["wind"])
            ),
            message_hi=(
                "तेज हवा का पूर्वानुमान: {:.1f} मीटर/सेकंड। "
                "आँधी से फसल स्वास्थ्य प्रभावित होने का खतरा।".format(w["wind"])
            ),
            action_en=(
                "Avoid spraying operations — spray drift will render application ineffective. "
                "Stake tall crops (maize, sugarcane) to prevent lodging. "
                "Secure plastic sheets in poly-houses and nurseries."
            ),
            action_hi=(
                "छिड़काव कार्य न करें — हवा में बहाव से आवेदन अप्रभावी हो जाएगा। "
                "ऊंची फसलों (मक्का, गन्ना) को गिरने से बचाने के लिए खूंटी लगाएं। "
                "पॉलीहाउस एवं नर्सरी में प्लास्टिक शीट को सुरक्षित करें।"
            ),
            triggered_by={"wind_ms": w["wind"]},
        )
    return None


def _rule_moderate_wind_spray(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Wind 8–12 m/s — moderate wind, spray with caution."""
    if 8.0 < w["wind"] <= 12.0:
        return AdvisoryAlert(
            alert_id="WIND_MODERATE_SPRAY_CAUTION",
            category="Wind",
            severity="MODERATE",
            message_en=(
                "Moderate winds: {:.1f} m/s. "
                "Spray drift will reduce pesticide efficacy.".format(w["wind"])
            ),
            message_hi=(
                "मध्यम हवा: {:.1f} मीटर/सेकंड। "
                "हवा का बहाव कीटनाशक प्रभावकारिता कम करेगा।".format(w["wind"])
            ),
            action_en=(
                "Spray only during early morning or late evening when winds are calm (<4 m/s). "
                "Use drift-reducing nozzles. Avoid spraying near water bodies."
            ),
            action_hi=(
                "केवल सुबह जल्दी या शाम को जब हवा शांत हो (<4 मीटर/सेकंड) छिड़काव करें। "
                "बहाव-रोधी नोजल का उपयोग करें। जल निकायों के पास छिड़काव से बचें।"
            ),
            triggered_by={"wind_ms": w["wind"]},
        )
    return None


def _rule_composite_waterlogging(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Rainfall > 20 mm + RH > 85% → combined waterlogging + disease risk."""
    if w["rainfall"] > 20.0 and w["rh"] > 85.0:
        return AdvisoryAlert(
            alert_id="COMPOSITE_WATERLOG_DISEASE",
            category="Composite",
            severity="HIGH",
            message_en=(
                "Combined heavy rainfall ({:.0f} mm) and very high humidity ({:.0f}%) "
                "create conditions for both waterlogging and soil-borne disease outbreaks "
                "(Pythium, Phytophthora root rots).".format(w["rainfall"], w["rh"])
            ),
            message_hi=(
                "भारी वर्षा ({:.0f} मिमी) एवं अत्यधिक आर्द्रता ({:.0f}%) "
                "जलभराव और मृदाजनित रोग (पाइथियम, फाइटोफ्थोरा जड़ सड़न) "
                "दोनों के लिए खतरनाक परिस्थितियाँ।".format(w["rainfall"], w["rh"])
            ),
            action_en=(
                "Open field channels for drainage immediately. "
                "Apply Metalaxyl-M 4% + Mancozeb 64% WP as soil drench near root zone. "
                "Do not apply nitrogen fertilizer during waterlogging."
            ),
            action_hi=(
                "जल निकास के लिए खेत की नालियां तुरंत खोलें। "
                "जड़ क्षेत्र के पास मेटालेक्सिल-M 4% + मैन्कोजेब 64% WP का मृदा उपचार करें। "
                "जलभराव के दौरान नाइट्रोजन उर्वरक न दें।"
            ),
            triggered_by={"rainfall_mm": w["rainfall"], "rh_pct": w["rh"]},
        )
    return None


def _rule_dry_spell_irrigation(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """Rainfall < 2 mm + Tmax > 38°C → dry spell, irrigation advisory."""
    if w["rainfall"] < 2.0 and w["tmax"] > 38.0:
        return AdvisoryAlert(
            alert_id="IRRIGATION_DRY_SPELL",
            category="Irrigation",
            severity="MODERATE",
            message_en=(
                "Dry conditions (rainfall < 2 mm) with high temperature ({:.1f}°C). "
                "Risk of drought stress in standing crops.".format(w["tmax"])
            ),
            message_hi=(
                "शुष्क परिस्थितियां (वर्षा < 2 मिमी) एवं उच्च तापमान ({:.1f}°C)। "
                "खड़ी फसलों में सूखे का तनाव।".format(w["tmax"])
            ),
            action_en=(
                "Irrigate crops immediately. Prioritize critical growth stages "
                "(flowering and grain-fill for cereals; flowering and pod-fill for legumes). "
                "Consider micro-irrigation to conserve water."
            ),
            action_hi=(
                "तुरंत फसलों की सिंचाई करें। "
                "महत्वपूर्ण वृद्धि अवस्थाओं (अनाज में पुष्पन एवं दाना भरना; दलहन में "
                "पुष्पन एवं फली भरना) को प्राथमिकता दें। "
                "जल संरक्षण के लिए सूक्ष्म सिंचाई पर विचार करें।"
            ),
            triggered_by={"rainfall_mm": w["rainfall"], "tmax_c": w["tmax"]},
        )
    return None


def _rule_favourable_conditions(w: Dict[str, float]) -> Optional[AdvisoryAlert]:
    """
    Catch-all: when no stress conditions are detected, issue a positive advisory.
    Only fires if no other rule has matched.
    """
    # This is a sentinel rule — the engine only adds it if no other alert fired
    return AdvisoryAlert(
        alert_id="CONDITIONS_FAVOURABLE",
        category="General",
        severity="LOW",
        message_en=(
            "Weather conditions are generally favourable for crop growth. "
            "Tmax={:.1f}°C, Rainfall={:.1f}mm, RH={:.0f}%.".format(
                w["tmax"], w["rainfall"], w["rh"])
        ),
        message_hi=(
            "मौसम परिस्थितियां फसल वृद्धि के लिए सामान्यतः अनुकूल हैं। "
            "अधिकतम तापमान={:.1f}°C, वर्षा={:.1f}मिमी, आर्द्रता={:.0f}%।".format(
                w["tmax"], w["rainfall"], w["rh"])
        ),
        action_en=(
            "Continue regular crop monitoring. "
            "Carry out scheduled fertilization and inter-cultural operations."
        ),
        action_hi=(
            "नियमित फसल निगरानी जारी रखें। "
            "निर्धारित उर्वरक प्रयोग एवं अंतःसस्य क्रियाएं करें।"
        ),
        triggered_by={
            "tmax_c": w["tmax"],
            "rainfall_mm": w["rainfall"],
            "rh_pct": w["rh"],
        },
    )


# ---------------------------------------------------------------------------
# Rule registry — ordered from most to least severe
# ---------------------------------------------------------------------------

RULE_REGISTRY: List[Callable[[Dict[str, float]], Optional[AdvisoryAlert]]] = [
    _rule_heavy_rainfall,
    _rule_extreme_heat,
    _rule_composite_waterlogging,
    _rule_cold_wave,
    _rule_high_wind_dust,
    _rule_high_rh_fungal,
    _rule_moderate_rainfall_spray,
    _rule_very_high_rh_avoid_ops,
    _rule_moderate_heat,
    _rule_moderate_wind_spray,
    _rule_dry_spell_irrigation,
    _rule_light_rainfall_sow,
]


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_advisory(
    weather: Dict[str, float],
    panchayat_id: str = "UNKNOWN",
    max_alerts: int = 5,
) -> List[Dict[str, Any]]:
    """
    Evaluate all advisory rules against the given panchayat weather conditions.

    Parameters
    ----------
    weather : dict
        Keys: tmax (°C), tmin (°C), rainfall (mm/day), rh (%), wind (m/s).
    panchayat_id : str
        Gram Panchayat identifier (for logging / traceability).
    max_alerts : int
        Maximum number of alerts to return (highest severity first).

    Returns
    -------
    List[dict]
        JSON-serialisable list of alert dicts, sorted by severity descending.
        Always returns at least one alert (the favourable-conditions sentinel).
    """
    # Validate input keys
    required = {"tmax", "tmin", "rainfall", "rh", "wind"}
    missing = required - set(weather.keys())
    if missing:
        raise ValueError(f"Missing weather variables: {missing}")

    fired: List[AdvisoryAlert] = []

    for rule_fn in RULE_REGISTRY:
        alert = rule_fn(weather)
        if alert is not None:
            fired.append(alert)

    # If no rules matched, emit the favourable-conditions sentinel
    if not fired:
        fired.append(_rule_favourable_conditions(weather))

    # Sort by severity (CRITICAL first) and cap
    fired.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 0), reverse=True)
    fired = fired[:max_alerts]

    # Attach metadata
    timestamp = datetime.datetime.now(timezone.utc).isoformat()
    return [
        {
            "panchayat_id": panchayat_id,
            "generated_at": timestamp,
            **alert.to_dict(),
        }
        for alert in fired
    ]


# ---------------------------------------------------------------------------
# Convenience: batch evaluation over multiple panchayats
# ---------------------------------------------------------------------------

def evaluate_batch(
    panchayat_weather_map: Dict[str, Dict[str, float]],
    max_alerts_per_panchayat: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate advisories for multiple panchayats at once.

    Parameters
    ----------
    panchayat_weather_map : dict
        {panchayat_id: weather_dict}

    Returns
    -------
    dict
        {panchayat_id: [alert_dicts]}
    """
    return {
        pid: evaluate_advisory(w, panchayat_id=pid, max_alerts=max_alerts_per_panchayat)
        for pid, w in panchayat_weather_map.items()
    }


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = [
        {
            "label": "Heavy monsoon rain + high humidity (Composite worst-case)",
            "pid": "RJ_CH_001",
            "weather": {"tmax": 33.0, "tmin": 26.0, "rainfall": 62.0, "rh": 91.0, "wind": 4.5},
        },
        {
            "label": "Summer heat wave (dry spell, no rain)",
            "pid": "RJ_CH_002",
            "weather": {"tmax": 44.5, "tmin": 31.0, "rainfall": 0.0, "rh": 22.0, "wind": 9.0},
        },
        {
            "label": "Moderate rain + fungal risk",
            "pid": "RJ_CH_003",
            "weather": {"tmax": 35.0, "tmin": 24.0, "rainfall": 28.0, "rh": 85.0, "wind": 3.2},
        },
        {
            "label": "Favourable conditions",
            "pid": "RJ_CH_004",
            "weather": {"tmax": 34.0, "tmin": 22.0, "rainfall": 10.0, "rh": 65.0, "wind": 4.0},
        },
    ]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario : {scenario['label']}")
        print(f"Panchayat: {scenario['pid']}")
        alerts = evaluate_advisory(scenario["weather"], panchayat_id=scenario["pid"])
        for a in alerts:
            print(f"  [{a['severity']:8s}] [{a['category']:12s}] {a['alert_id']}")
            print(f"    EN: {a['message_en'][:80]}...")
            hi_safe = a['message_hi'][:60].encode('ascii', errors='replace').decode('ascii')
            print(f"    HI(ascii): {hi_safe}...")

