from medical_frame import MedicalFrame
from appointment_frame import AppointmentFrame
from general_query_handler import GeneralQueryFrame
import joblib 
import re
import spacy
from spacy.matcher import PhraseMatcher


class BaseFrame:
    def __init__(self):
        self.slots = {}

    def update_slots(self, text):
        pass

    def is_complete(self):
        pass

    def reset_slots(self):
        for key in self.slots.keys():
            self.slots[key] = None

    def get_missing_slots(self):
        pass



nlp = spacy.load("en_core_web_md")

# Initialize PhraseMatcher with the English vocabulary
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

appointment_phrases = [
    "book", "schedule", "a visit", "reserve","a consultation", "make","appointment","booking"
    "set up","checkup", "arrange","meeting","a meeting with doctor", "organize a session", "secure a slot",
    "register for an appointment", "fix a date", "plan","a visit", "book a time","get a"
    "get a slot", "enroll for a consultation", "request an appointment", "book a medical check",
    "schedule a physical", "reserve a medical session", "book a doctor's appointment",
    "schedule a treatment session", "arrange a health check", "follow-up","doctors"
    "secure a medical examination", "confirm","an appointment", "place an appointment",
    "book a slot","slot" "appoint a visit", "sign up for consultation", "set an appointment",
    "schedule an operation", "reserve an examination", "request a time slot",
    "make a reservation", "book a medical review", "arrange a surgery time",
    "set a doctor's visit", "organize a clinical assessment", "book an ","emergency", "appointment",
    "schedule","dental check", "secure an appointment", "specialist"
]


booking_patterns = [nlp.make_doc(text.lower()) for text in appointment_phrases]  
matcher.add("BOOKING_APPOINTMENT", booking_patterns)


symptom_patterns = [ "Abdominal Pain", "Acid Reflux", "Airsickness", "Bad Breath", "Belching",
    "Bellyache", "Bleeding", "Bleeding, Gastrointestinal", "Breath Odor",
    "Breathing Problems", "Bruises", "Burping", "Carsickness", "Chest Pain",
    "Chilblains", "Choking", "Chronic Pain", "Cluster Headache", "Cold (Temperature)",
    "Communication Disorders", "Constipation", "Contusions", "Cough", "Dehydration",
    "Diarrhea", "Dizziness and Vertigo", "Dropsy", "Dysentery", "Dysfunctional uterine bleeding",
    "Dyspepsia", "Dyspnea", "Edema", "Fainting", "Fatigue", "Fever", "Flatulence",
    "Frostbite", "Frostnip", "Gas", "Gastrointestinal Bleeding", "GI Bleeding","blood","bleeding",
    "Halitosis", "Headache", "Heartburn", "Heat Exhaustion", "Heat Illness","breath","eyes",
    "Heimlich Maneuver", "Hematoma", "Hemorrhage", "Hives", "Hot (Temperature)",
    "Hypothermia", "Icterus", "Indigestion", "Itching", "Jaundice", "Kernicterus",
    "Language Problems", "Motion Sickness", "Nausea and Vomiting", "Pain", "Pain, Abdominal",
    "Pain", "Chest", "Pain", "Chronic", "Pelvic Pain", "Pruritus", "Pyrexia", "Rare Diseases",
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





patterns_sym = [nlp.make_doc(text) for text in symptom_patterns]
matcher.add("SYMPTOM", patterns_sym)


query_patterns = [nlp(text) for text in ['how', 'what', 'where', 'when', 'why']]
matcher.add('GENERAL_QUERY', query_patterns)

class DialogManager:
    def __init__(self, model_path):
        self.initial_greeting = True 
        self.recommendation_given = False 
        self.current_frame = None
        self.intent = ""
        if model_path:
            self.model, self.vectorizer = joblib.load(model_path)

    def greet(self):
        return "I'm your medical assistant, how can I help you?"

    def process_input(self, user_input):
        if self.initial_greeting:
            self.initial_greeting = False
            return self.greet()
        
        # if self.recommendation_given:
            
        #     return "What else can I do for you? If you liked my service, please provide feedback."

        # Intent recognition needs to happen here
        if  not self.intent: 
            self.intent = self.detect_intent(user_input)
            self.set_current_frame(self.intent)

        self.current_frame.update_slots(user_input)
        
        if self.current_frame.is_complete():
            return self.provide_recommendation()

        else:
            return self.ask_for_missing_info()

    def detect_intent(self, text):
        doc = nlp(text)
        matches = matcher(doc)

        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]  
            if rule_id == 'BOOKING_APPOINTMENT':
                return "booking_appointment"
            elif rule_id == 'SYMPTOM':
                return "symptom_description"

        return "general_query"


    def set_current_frame(self, intent):
        if intent == "booking_appointment":
            self.current_frame = AppointmentFrame()
        elif intent == "symptom_description":
            self.current_frame = MedicalFrame()
        else:
            self.current_frame = GeneralQueryFrame()
            
            
    def ask_for_missing_info(self):
        return self.current_frame.get_missing_slots()
    
    def provide_recommendation(self):
        frame_response = self.current_frame.generate_response()
        self.current_frame.reset_slots()
        self.intent =""
        self.recommendation_given = True
        self.initial_greeting = False 
        return frame_response


# Assuming model path is provided correctly
dm = DialogManager("trained_model.joblib")
# print(dm.process_input(" fever and cold"))
# print(dm.process_input(" my name is rajesh want to book the appointment at 17 july 2024 evening "))
# print(dm.process_input(" 17 july 2024 evening"))
# print(dm.process_input("july 6th evening 7:30pm"))
# print(dm.process_input(" I've been having headachesIt started a few days agoThe pain is quite severe I haven't taken any medication yet"))

