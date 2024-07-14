import re
import spacy
from spacy.matcher import PhraseMatcher
import joblib

nlp = spacy.load("en_core_web_md")  # Load spaCy model

# Create a PhraseMatcher object
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Add custom symptom patterns to the matcher
symptom_patterns = [ "Abdominal Pain", "Acid Reflux", "Airsickness", "Bad Breath", "Belching",
    "Bellyache", "Bleeding", "Bleeding, Gastrointestinal", "Breath Odor",
    "Breathing Problems", "Bruises", "Burping", "Carsickness", "Chest Pain",
    "Chilblains", "Choking", "Chronic Pain", "Cluster Headache", "Cold (Temperature)",
    "Communication Disorders", "Constipation", "Contusions", "Cough", "Dehydration",
    "Diarrhea", "Dizziness and Vertigo", "Dropsy", "Dysentery", "Dysfunctional uterine bleeding",
    "Dyspepsia", "Dyspnea", "Edema", "Fainting", "Fatigue", "Fever", "Flatulence",
    "Frostbite", "Frostnip", "Gas", "Gastrointestinal Bleeding", "GI Bleeding",
    "Halitosis", "Headache", "Heartburn", "Heat Exhaustion", "Heat Illness",
    "Heimlich Maneuver", "Hematoma", "Hemorrhage", "Hives", "Hot (Temperature)",
    "Hypothermia", "Icterus", "Indigestion", "Itching", "Jaundice", "Kernicterus",
    "Language Problems", "Motion Sickness", "Nausea and Vomiting", "Pain", "Pain, Abdominal",
    "Pain, Chest", "Pain, Chronic", "Pelvic Pain", "Pruritus", "Pyrexia", "Rare Diseases",
    "Raynaud Phenomenon", "Raynaud's Disease", "Raynaud's Syndrome", "Rectal Bleeding",
    "Sciatica", "Seasickness", "Shortness of Breath", "Speech and Communication Disorders",
    "Stammering", "Stomach Ache", "Stuttering", "Sunstroke", "Swelling", "Syncope",
    "Tachypnea", "Tension Headache", "Thirst", "Tiredness", "Upset Stomach", "Urticaria",
    "Uterine bleeding", "Vaginal Bleeding", "Vascular Headache", "Vasovagal Syncope",
    "Vertigo", "Vestibular Diseases", "Vomiting", "Weariness",
    "headache", "vomiting", "indigestion", "loss of appetite", "abdominal pain",
    "passage of gases", "internal itching", "pain", "severe pain", "mild pain", "moderate pain",
    "fever", "cough", "sore throat", "runny nose", "fatigue", "nausea", "dizziness",
    "shortness of breath", "chest pain", "diarrhea", "constipation", "back pain", "muscle pain",
    "joint pain", "rash", "swelling", "bruising", "itching", "burning sensation", "tingling",
    "numbness", "weakness", "palpitations", "anxiety", "depression", "insomnia", "weight loss",
    "weight gain", "hair loss", "blurred vision", "dry eyes", "ringing in ears", "ear pain",
    "nosebleed", "difficulty swallowing", "heartburn", "bloating", "urinary frequency", "blood in urine"
]




# Create pattern docs and add them to the matcher
patterns = [nlp.make_doc(text) for text in symptom_patterns]
matcher.add("SYMPTOM", patterns)

class MedicalFrame:
    def __init__(self):
        self.reset_slots()
        
    def reset_slots(self):
        self.model, self.vectorizer = joblib.load("trained_model.joblib")
        self.slots = {
            "Symptoms": None,
            "Duration": None,
            "Severity": None
        }

    def update_slots(self, text):
        doc = nlp(text)
        matches = matcher(doc)  # Use the matcher to find symptoms in the text
        symptoms = []

        for match_id, start, end in matches:
            span = doc[start:end]  # The matched span
            symptoms.append(span.text)

        # Update slots based on the findings
        if symptoms:
            self.slots["Symptoms"] = ", ".join(symptoms)
        
        for ent in doc.ents:
            if ent.label_ == "DATE" and not self.slots["Duration"]:
                self.slots["Duration"] = ent.text  # Capturing date as part of duration
            elif ent.label_ == "TIME" and not self.slots["Duration"]:
                # Capturing time if it hasn't been captured as part of the duration yet
                self.slots["Duration"] = ent.text
            elif ent.label_ == "DURATION" and not self.slots["Duration"]:
                # Directly capturing duration if identified by SpaCy
                self.slots["Duration"] = ent.text
            elif ent.label_ == "CARDINAL" and 'day' in text.lower():
                self.slots["Duration"] = ent.text + " days"

        # Severity handling
        # severity_levels = {"severe", "moderate", "mild"}
        # found_severities = [word for word in severity_levels if word in text.lower()]
        # if found_severities:
        #     self.slots["Severity"] = found_severities[0]
        severity_levels = {"severe", "moderate", "mild"}
        found_severities = [word for word in severity_levels if word in text.lower()]
        if found_severities:
            self.slots["Severity"] = found_severities[0]
        else:
            # Check for numerical severity on a scale from 1 to 10
            numbers = re.findall(r'\b([1-9]|10)\b', text)
            if numbers:
                num = int(numbers[0])
                if 1 <= num <= 3:
                    self.slots["Severity"] = " Severity is mild"
                elif 4 <= num <= 6:
                    self.slots["Severity"] = "Severity is moderate"
                else:
                    self.slots["Severity"] = "Severity is high"

    def is_complete(self):
        return all(value is not None for value in self.slots.values())

    def get_missing_slots(self):
        
        missing_slots = [slot for slot, value in self.slots.items() if value is None]
        if not missing_slots:
            return "Please contact the doctor for immediate assistance!"

        slot_question_map = {
            "Symptoms": "Can you describe your symptoms in more detail?",
            "Duration": "How long have you been experiencing these symptoms?",
            "Severity": "Could you rate the severity of your symptoms on a scale from 1 to 10?",
            "Previous_Treatment": "Have you received any treatment or taken any medication for these symptoms?",
            "Date": "At what date did you required to book the appointment?",
            "Time":"can you specify the time for booking?"
        }

        # Ask about the first missing slot found
        for slot in missing_slots:
            if slot in slot_question_map:
                return slot_question_map[slot]

        return "Can you provide more details?"
        # return [slot for slot, value in self.slots.items() if value is None]
    def generate_response(self):
           # If doctor is not yet set, choose one at random
            # combined_text = " ".join([v for k, v in self.slots.items() if v is not None])
            features = " ".join([v for k, v in self.slots.items() if v is not None and k == 'Symptoms'])
            # features = self.vectorizer.transform([self.slots["Symptoms"]])
            combined_text_vectorized = self.vectorizer.transform([features])
            prediction = self.model.predict(combined_text_vectorized)

            if prediction:
                self.recommendation_given = True
                return f"Based on your symptoms, you might be experiencing {prediction[0]}. If you need further assistance, please contact a doctor."
            else:
                return "Please contact a doctor for immediate assistance!"