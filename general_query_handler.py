import spacy
from spacy.matcher import Matcher
import random

nlp = spacy.load("en_core_web_md")


clinic_data = {
    "doctors": {
        "Dr. Parvathi": {"specialty": "cardiologist", "experience": 20, "hours": "9 AM to 5 PM"},
        "Dr. Madhumitha": {"specialty": "neurologist", "experience": 15, "hours": "10 AM to 6 PM"}
    },
    "services": ["cardiology", "neurology", "pediatric care", "emergency care"],
    "hours": "Open daily from 8 AM to 8 PM",
    "location": "cubbon park street., Bangalore-530068"
}

class GeneralQueryFrame:
    def __init__(self):
        self.slots = {
            "query": None
        }
        self.text =''
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab)
        
        query_patterns = [
            [{"LOWER": "who"}, {"LOWER": "is"}, {"LOWER": "the"}, {"LOWER": "best"}, {"LOWER": "doctor"}],
            [{"LOWER": "who"}, {"LOWER": "is"}, {"LOWER": "the"}, {"LOWER": "most"}, {"LOWER": "famous"}, {"LOWER": "doctor"}],
            [{"LOWER": "what"}, {"LOWER": "are"}, {"LOWER": "your"}, {"LOWER": "hours"}],
            [{"LOWER": "what"}, {"LOWER": "time"}, {"LOWER": "do"}, {"LOWER": "you"}, {"LOWER": "open"}],
            [{"LOWER": "can"}, {"LOWER": "you"}, {"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "about"}, {"LOWER": "the"}, {"LOWER": "services"}, {"LOWER": "you"}, {"LOWER": "offer"}],
            [{"LOWER": "where"}, {"LOWER": "is"}, {"LOWER": "the"}, {"LOWER": "clinic"}],
            [{"LOWER": "what"}, {"LOWER": "services"}, {"LOWER": "do"}, {"LOWER": "you"}, {"LOWER": "provide"}]
        ]

        for pattern in query_patterns:
            self.matcher.add('GENERAL_QUERY', [pattern])

    def process_query(self, text):
        self.text = text
        doc = self.nlp(text)
        matches = self.matcher(doc)
        if matches:
            return self.answer_query(text)
        else:
            return "I'm sorry, I didn't understand your question."

    def answer_query(self, query):
        query = query.lower()
        if "famous doctor" in query or "best doctor" in query:
            most_experienced = max(clinic_data["doctors"].items(), key=lambda x: x[1]['experience'])
            doctor, details = most_experienced
            return f"{doctor} is the most experienced, specializing as a {details['specialty']} with {details['experience']} years of experience."
        elif "open" in query or "hours" in query:
            return f"The clinic is {clinic_data['hours']}."
        elif "services" in query:
            services = ", ".join(clinic_data["services"])
            return f"We offer the following services: {services}."
        elif "location" in query:
            return f"We are located at {clinic_data['location']}."
        else:
            return "I'm sorry, I didn't understand your question.  I can Help you in disease recognition, Appointment Booking & general info about the hospital!"
    
    def is_complete(self):
        return True
    def reset_slots(self):
        self.text = ""
    def generate_response(self):
        return self.answer_query(self.text)
    def update_slots(self, text):
        self.text = text
        pass


# Example of using the class
# gqf = GeneralQueryFrame()
# print(gqf.process_query("Who is the best doctor?"))
# print(gqf.process_query("What are your hours?"))
# print(gqf.process_query("What services do you provide?"))
# print(gqf.process_query("Where is the clinic located?"))
# print(gqf.is_complete())
# print(gqf.generate_response())

