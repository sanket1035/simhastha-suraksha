"""Words for the public pilgrim page in English, Hindi and Marathi.

Short, plain sentences. The control room stays in English. Marathi and Hindi wording should be checked by a
native speaker before a real deployment.
"""
LANGS = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}
CODES = {v: k for k, v in LANGS.items()}

ZONE_NAMES = {"Ramkund": {"hi": "रामकुंड", "mr": "रामकुंड"},
              "Panchvati": {"hi": "पंचवटी", "mr": "पंचवटी"},
              "Trimbakeshwar": {"hi": "त्र्यंबकेश्वर", "mr": "त्र्यंबकेश्वर"}}

LEVEL_FULL = {
    "en": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"},
    "hi": {"Low": "भीड़ कम", "Medium": "भीड़ मध्यम", "High": "भीड़ ज्यादा", "Critical": "बहुत भीड़"},
    "mr": {"Low": "गर्दी कमी", "Medium": "गर्दी मध्यम", "High": "गर्दी जास्त", "Critical": "खूप गर्दी"},
}
LEVEL_SHORT = {
    "en": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"},
    "hi": {"Low": "कम", "Medium": "मध्यम", "High": "ज्यादा", "Critical": "बहुत ज्यादा"},
    "mr": {"Low": "कमी", "Medium": "मध्यम", "High": "जास्त", "Critical": "खूप जास्त"},
}
ADVICE = {
    "en": {"Low": "Clear. Good time to visit.",
           "Medium": "Moderately busy. Keep moving and stay with your group.",
           "High": "Crowded. Consider waiting or using another ghat.",
           "Critical": "Very crowded. Please avoid this area now and follow police directions."},
    "hi": {"Low": "भीड़ कम है। दर्शन के लिए अच्छा समय है।",
           "Medium": "थोड़ी भीड़ है। चलते रहें और अपने साथियों के साथ रहें।",
           "High": "भीड़ ज्यादा है। थोड़ा रुकें या दूसरे घाट पर जाएं।",
           "Critical": "बहुत भीड़ है। अभी इस जगह न जाएं और पुलिस की बात मानें।"},
    "mr": {"Low": "गर्दी कमी आहे. दर्शनासाठी चांगली वेळ आहे.",
           "Medium": "थोडी गर्दी आहे. चालत राहा आणि आपल्या सोबत्यांसोबत राहा.",
           "High": "गर्दी जास्त आहे. थोडं थांबा किंवा दुसऱ्या घाटावर जा.",
           "Critical": "खूप गर्दी आहे. आत्ता या ठिकाणी जाऊ नका आणि पोलिसांचं ऐका."},
}
TEXT = {
    "en": {"top": "Simhastha Kumbh Mela · Nashik", "kicker": "Live crowd status",
           "blurb": "Know how busy each ghat is before you go. Updated from the cameras, in plain words.",
           "where": "Where is it crowded", "close": "Look closer",
           "updated": "Updated %s · %d camera angle%s", "waiting": "Waiting for the first camera update.",
           "none_t": "No live data yet", "none_b": "Crowd updates appear here as soon as the cameras report.",
           "guide": "Green = low · Yellow = medium · Orange = high · Red = very crowded",
           "foot": "This page refreshes by itself. Always follow the instructions of police and volunteers on site.",
           "ends": ("few people", "some", "crowded"), "nodata": "No data yet",
           "info": "%s · %s people · %.2f per m²", "lang": "Language",
           "views": {"Heatmap overlay": "Heatmap overlay", "Heat only": "Heat only",
                     "On the map": "On the map", "Original": "Original"},
           "stale": "This view needs a fresh photo; upload again to draw the map."},
    "hi": {"top": "सिंहस्थ कुंभ मेला · नाशिक", "kicker": "लाइव भीड़ की स्थिति",
           "blurb": "जाने से पहले जानें कि हर घाट पर कितनी भीड़ है। कैमरों से मिली जानकारी, सरल शब्दों में।",
           "where": "कहाँ भीड़ है", "close": "नज़दीक से देखें",
           "updated": "अपडेट: %s · %d कैमरा", "waiting": "पहले कैमरा अपडेट का इंतज़ार है।",
           "none_t": "अभी कोई लाइव जानकारी नहीं", "none_b": "कैमरे की जानकारी आते ही यहाँ दिखेगी।",
           "guide": "हरा = कम · पीला = मध्यम · नारंगी = ज्यादा · लाल = बहुत भीड़",
           "foot": "यह पेज अपने आप अपडेट होता है। हमेशा मौके पर पुलिस और स्वयंसेवकों की बात मानें।",
           "ends": ("कम लोग", "कुछ", "भीड़"), "nodata": "अभी जानकारी नहीं",
           "info": "%s · %s लोग · प्रति m² %.2f", "lang": "भाषा",
           "views": {"Heatmap overlay": "फोटो पर भीड़", "Heat only": "सिर्फ़ भीड़ का रंग",
                     "On the map": "नक्शे पर", "Original": "असली फोटो"},
           "stale": "इस व्यू के लिए नया फोटो चाहिए।"},
    "mr": {"top": "सिंहस्थ कुंभमेळा · नाशिक", "kicker": "लाईव्ह गर्दीची स्थिती",
           "blurb": "जाण्यापूर्वी जाणून घ्या की प्रत्येक घाटावर किती गर्दी आहे. कॅमेऱ्यांवरून मिळालेली माहिती, सोप्या शब्दांत.",
           "where": "कुठे गर्दी आहे", "close": "जवळून पाहा",
           "updated": "अपडेट: %s · %d कॅमेरा", "waiting": "पहिल्या कॅमेरा अपडेटची वाट पाहत आहे.",
           "none_t": "अजून लाईव्ह माहिती नाही", "none_b": "कॅमेऱ्याची माहिती येताच इथे दिसेल.",
           "guide": "हिरवा = कमी · पिवळा = मध्यम · नारिंगी = जास्त · लाल = खूप गर्दी",
           "foot": "हे पेज आपोआप अपडेट होते. नेहमी पोलीस आणि स्वयंसेवकांच्या सूचनांचे पालन करा.",
           "ends": ("कमी लोक", "काही", "गर्दी"), "nodata": "अजून माहिती नाही",
           "info": "%s · %s लोक · प्रति m² %.2f", "lang": "भाषा",
           "views": {"Heatmap overlay": "फोटोवर गर्दी", "Heat only": "फक्त गर्दीचा रंग",
                     "On the map": "नकाशावर", "Original": "मूळ फोटो"},
           "stale": "या व्ह्यूसाठी नवीन फोटो हवा आहे."},
}


def zone_name(zone, lang):
    return ZONE_NAMES.get(zone, {}).get(lang, zone)


def t(lang, key):
    return TEXT.get(lang, TEXT["en"])[key]
