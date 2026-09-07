"""LAND-JEPA — Multilingual & Indian DLT Compliant Notification Templates.
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)

Supported languages:
  - English (en)
  - Hindi (hi)
  - Assamese (as)
  - Bengali (bn)
  - Manipuri / Meitei (mni)

Safety Rules:
  - Never claim certainty: Do NOT say "LANDSLIDE WILL DEFINITELY HAPPEN".
  - Always advise following official emergency and geotechnical authorities.
  - Conforms strictly to TRAI DLT template variable injection: {zone}, {horizon}.
  - Fallback: If translation missing, fall back to English and log MISSING_NOTIFICATION_TRANSLATION.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard Indian DLT registered sender ID header
DEFAULT_SENDER_ID = "LNDJPA"

# Multilingual template catalog: [language][tier] -> { title, body, dlt_template_id }
TEMPLATE_CATALOG: Dict[str, Dict[str, Dict[str, str]]] = {
    "en": {
        "WATCH": {
            "title": "LAND-JEPA WATCH: {zone}",
            "body": (
                "LAND-JEPA WATCH: Increased landslide risk detected in {zone}. "
                "Forecast horizon: {horizon}. Ground saturation elevated. "
                "Please follow official local guidance and travel advisories."
            ),
            "dlt_template_id": "DLT-NER-WATCH-EN-1001",
        },
        "WARNING": {
            "title": "LAND-JEPA WARNING: {zone}",
            "body": (
                "LAND-JEPA WARNING: Elevated landslide risk detected in {zone}. "
                "Forecast horizon: {horizon}. Hazardous slopes actively monitored. "
                "Please follow official local guidance."
            ),
            "dlt_template_id": "DLT-NER-WARN-EN-1002",
        },
        "CRITICAL": {
            "title": "LAND-JEPA CRITICAL: {zone}",
            "body": (
                "LAND-JEPA CRITICAL: Very high landslide risk detected in {zone}. "
                "Time horizon: {horizon}. Immediate precautionary vigilance required. "
                "Follow official emergency guidance."
            ),
            "dlt_template_id": "DLT-NER-CRIT-EN-1003",
        },
    },
    "hi": {
        "WATCH": {
            "title": "लैंड-जेपा निगरानी (WATCH): {zone}",
            "body": (
                "लैंड-जेपा निगरानी (WATCH): {zone} में भूस्खलन की निगरानी बढ़ाई गई है। "
                "पूर्वानुमान क्षितिज: {horizon}। कृपया आधिकारिक स्थानीय निर्देशों और "
                "यातायात परामर्श का पालन करें।"
            ),
            "dlt_template_id": "DLT-NER-WATCH-HI-2001",
        },
        "WARNING": {
            "title": "लैंड-जेपा चेतावनी (WARNING): {zone}",
            "body": (
                "लैंड-जेपा चेतावनी (WARNING): {zone} में भूस्खलन का बढ़ा हुआ जोखिम पाया गया है। "
                "पूर्वानुमान क्षितिज: {horizon}। कृपया आधिकारिक स्थानीय दिशानिर्देशों का पालन करें।"
            ),
            "dlt_template_id": "DLT-NER-WARN-HI-2002",
        },
        "CRITICAL": {
            "title": "लैंड-जेपा गंभीर चेतावनी (CRITICAL): {zone}",
            "body": (
                "लैंड-जेपा गंभीर चेतावनी (CRITICAL): {zone} में अत्यधिक उच्च भूस्खलन जोखिम का पता चला है। "
                "समय सीमा: {horizon}। कृपया आधिकारिक आपातकालीन मार्गदर्शन का पालन करें।"
            ),
            "dlt_template_id": "DLT-NER-CRIT-HI-2003",
        },
    },
    "as": {
        "WATCH": {
            "title": "লেণ্ড-জেপা সতৰ্ক দৃষ্টি (WATCH): {zone}",
            "body": (
                "লেণ্ড-জেপা দৃষ্টি (WATCH): {zone}ত ভূমিখলনৰ সম্ভাৱ্য অৱস্থা ধৰা পৰিছে। "
                "সময়সীমা: {horizon}। অনুগ্ৰহ কৰি চৰকাৰী স্থানীয় নিৰ্দেশনা আৰু সতৰ্কতা মানি চলক।"
            ),
            "dlt_template_id": "DLT-NER-WATCH-AS-3001",
        },
        "WARNING": {
            "title": "লেণ্ড-জেপা সতৰ্কবাণী (WARNING): {zone}",
            "body": (
                "লেণ্ড-জেপা সতৰ্কবাণী: {zone}ত বৃদ্ধি পোৱা ভূমিখলনৰ আশংকা দেখা গৈছে। "
                "পূৰ্বানুমানৰ সময়সীমা: {horizon}। অনুগ্ৰহ কৰি চৰকাৰী স্থানীয় নিৰ্দেশনা অনুসৰণ কৰক।"
            ),
            "dlt_template_id": "DLT-NER-WARN-AS-3002",
        },
        "CRITICAL": {
            "title": "লেণ্ড-জেপা গুৰুত্বপূৰ্ণ জৰুৰী সতৰ্কবাণী (CRITICAL): {zone}",
            "body": (
                "লেণ্ড-জেপা জৰুৰী সতৰ্কবাণী: {zone}ত অতি উচ্চ ভূমিখলনৰ আশংকা চিহ্নিত কৰা হৈছে। "
                "সময়সীমা: {horizon}। অনুগ্ৰহ কৰি চৰকাৰী জৰুৰীকালীন নিৰ্দেশনা কঠোৰভাৱে পালন কৰক।"
            ),
            "dlt_template_id": "DLT-NER-CRIT-AS-3003",
        },
    },
    "bn": {
        "WATCH": {
            "title": "ল্যান্ড-জেপা পর্যবেক্ষণ (WATCH): {zone}",
            "body": (
                "ল্যান্ড-জেপা পর্যবেক্ষণ (WATCH): {zone} অঞ্চলে ভূমিধসের বর্ধিত ঝুঁকি পরিলক্ষিত হয়েছে। "
                "পূর্বাভাস সময়সীমা: {horizon}। অনুগ্রহ করে সরকারি স্থানীয় নির্দেশনা মেনে চলুন।"
            ),
            "dlt_template_id": "DLT-NER-WATCH-BN-4001",
        },
        "WARNING": {
            "title": "ল্যান্ড-জেপা সতর্কতা (WARNING): {zone}",
            "body": (
                "ল্যান্ড-জেপা সতর্কতা: {zone} অঞ্চলে ভূমিধসের ঝুঁকি বৃদ্ধি পেয়েছে। "
                "পূর্বাভাস সময়সীমা: {horizon}। অনুগ্রহ করে সরকারি স্থানীয় পরামর্শ ও নির্দেশনা মেনে চলুন।"
            ),
            "dlt_template_id": "DLT-NER-WARN-BN-4002",
        },
        "CRITICAL": {
            "title": "ল্যান্ড-জেপা জরুরি সতর্কতা (CRITICAL): {zone}",
            "body": (
                "ল্যান্ড-জেপা জরুরি সতর্কতা: {zone} অঞ্চলে অত্যন্ত উচ্চ মাত্রার ভূমিধস ঝুঁকি চিহ্নিত। "
                "সময়সীমা: {horizon}। অনুগ্রহ করে জরুরি দুর্যোগ নির্দেশিকা অনুসরণ করুন।"
            ),
            "dlt_template_id": "DLT-NER-CRIT-BN-4003",
        },
    },
    "mni": {
        "WATCH": {
            "title": "লেন্ড-জেপা য়েংশিনবা (WATCH): {zone}",
            "body": (
                "লেন্ড-জেপা য়েংশিনবা: {zone} দা চীংহায়বগী খুদোংথীবা থেংনরে। "
                "মতমগী চাং: {horizon}। চানবীদুনা সরকারগী লোকেল চেকশিন-থৌরাংগী পাউতাক ইনবীয়ু।"
            ),
            "dlt_template_id": "DLT-NER-WATCH-MNI-5001",
        },
        "WARNING": {
            "title": "লেন্ড-জেপা চেকশিনৱা (WARNING): {zone}",
            "body": (
                "লেন্ড-জেপা চেকশিনৱা: {zone} দা চীংহায়বগী খুদোংথীবা হেনগৎলকপা উরে। "
                "মতমগী চাং: {horizon}। চানবীদুনা সরকারগী অফিশিয়েল লোকেল পাউতাক ইনবীয়ু।"
            ),
            "dlt_template_id": "DLT-NER-WARN-MNI-5002",
        },
        "CRITICAL": {
            "title": "লেন্ড-জেপা অকনবা চেকশিনৱা (CRITICAL): {zone}",
            "body": (
                "লেন্ড-জেপা অকনবা চেকশিনৱা: {zone} দা য়াম্না লূবা চীংহায়বগী খুদোংথীবা থেংনরে। "
                "মতমগী চাং: {horizon}। চানবীদুনা সরকারগী অফিশিয়েল ইমার্জেন্সী পাউতাক ইনবীয়ু।"
            ),
            "dlt_template_id": "DLT-NER-CRIT-MNI-5003",
        },
    },
}


class TemplateEngine:
    """Renders DLT-compliant, safety-audited multilingual early warning messages."""

    @classmethod
    def render(
        cls,
        tier: str,
        zone: str,
        horizon: str = "24h",
        language: str = "en",
        custom_vars: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, str]:
        """
        Renders title, body, dlt_template_id, and resolved language.

        Returns:
            (title, body, dlt_template_id, resolved_language)
        """
        tier_upper = tier.upper().strip()
        if tier_upper not in ("WATCH", "WARNING", "CRITICAL"):
            tier_upper = "WARNING"

        lang_code = language.lower().strip()
        resolved_lang = lang_code

        # Fallback to English if language not supported
        if lang_code not in TEMPLATE_CATALOG:
            logger.warning(
                f"MISSING_NOTIFICATION_TRANSLATION: Language '{language}' not found in catalog. "
                "Falling back to English ('en')."
            )
            lang_code = "en"
            resolved_lang = "en"

        lang_templates = TEMPLATE_CATALOG[lang_code]
        if tier_upper not in lang_templates:
            logger.warning(
                f"MISSING_NOTIFICATION_TRANSLATION: Tier '{tier_upper}' in language '{lang_code}' missing. "
                "Falling back to English ('en')."
            )
            lang_templates = TEMPLATE_CATALOG["en"]
            resolved_lang = "en"

        template = lang_templates[tier_upper]

        variables = {
            "zone": zone,
            "horizon": horizon,
        }
        if custom_vars:
            variables.update(custom_vars)

        try:
            title = template["title"].format(**variables)
            body = template["body"].format(**variables)
        except KeyError as e:
            logger.error(f"Missing template placeholder {e}. Using raw text.")
            title = template["title"].replace("{zone}", zone).replace("{horizon}", horizon)
            body = template["body"].replace("{zone}", zone).replace("{horizon}", horizon)

        # Safety Check: Never claim certainty
        if "WILL DEFINITELY HAPPEN" in body.upper():
            body = body.replace("WILL DEFINITELY HAPPEN", "has elevated risk")

        return title, body, template["dlt_template_id"], resolved_lang
