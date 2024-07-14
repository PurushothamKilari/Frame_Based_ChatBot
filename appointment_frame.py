import spacy
from spacy.matcher import Matcher
import random

nlp = spacy.load("en_core_web_md")  



name_patterns = [
    [{"LOWER": "my"}, {"LOWER": "name"}, {"LOWER": "is"}],
    [{"LOWER": "i'm"}],
    [{"LOWER": "i"}, {"LOWER": "am"}],
    [{"LOWER": "call"}, {"LOWER": "me"}],
    [{"TEXT": "Mr."}],
    [{"TEXT": "Mrs."}],
    [{"TEXT": "Miss"}],
    [{"LOWER": "boy"}, {"LOWER": "name"}],
    [{"LOWER": "girl"}, {"LOWER": "name"}],
]

matcher = Matcher(nlp.vocab)
for pattern in name_patterns:
    matcher.add("NAME_PATTERN", [pattern])

class AppointmentFrame:
    def __init__(self):
        super().__init__()

        self.slots = {
            "Date": None,
            "Time": None,
            "Person": None
        }
        self.mock_data = {
            "Dr. Smith": {"cost": "$100", "waiting_time": "15 minutes"},
            "Dr. Johnson": {"cost": "$150", "waiting_time": "5 minutes"}
        }

    def update_slots(self, text):
        doc = nlp(text)

        matches = matcher(doc)
        name_found = False

        for match_id, start, end in matches:
            if nlp.vocab.strings[match_id] == "NAME_PATTERN":
                # Assuming the name is the token right after the pattern ends
                if end < len(doc) and doc[end].is_alpha:
                    print(f"Name found: {doc[end].text}")
                    self.slots["Person"] = doc[end].text
                    name_found = True
                    break

        # If no name was found with the matcher, check with NER as a fallback
        if not name_found:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    print(f"Name found with NER: {ent.text}")
                    self.slots["Person"] = ent.text  
                    break
            for ent in doc.ents:
                if ent.label_ == "DATE" and not self.slots["Date"]:
                    self.slots["Date"] = ent.text
                elif ent.label_ == "TIME" and not self.slots["Time"]:
                    self.slots["Time"] = ent.text
                elif ent.label_ == "PERSON" and not self.slots["Person"]:
                    self.slots["Person"] = ent.text  


    def generate_response(self):
           # If doctor is not yet set, choose one at random
        self.slots['Doctor'] = random.choice(list(self.mock_data.keys()))
        
        if not all(self.slots.values()):
            return self.update_slots("")  # Re-check for missing information
        
        doctor_info = self.mock_data.get(self.slots["Doctor"], {})
        return (f"Appointment booked with {self.slots['Doctor']} on {self.slots['Date']} at {self.slots['Time']}. "
                f"Cost: {doctor_info.get('cost', 'N/A')}, Waiting time: {doctor_info.get('waiting_time', 'N/A')}.")
    
    def reset_slots(self):
        for key in self.slots.keys():
            self.slots[key] = None

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
                "Date": "At what date did you required to book the appointment? Please provide the date for your appointment.",
                "Time":"What time would you like your appointment?",
                "Doctor": "please select any specalist to book further?",
                "Person":"Could you, Please provide your name?"
            }

            # Ask about the first missing slot found
            for slot in missing_slots:
                if slot in slot_question_map:
                    return slot_question_map[slot]

            return "Can you provide more details?"



# af = AppointmentFrame()
# af.update_slots("my name is rajesh want to book the appointment at 17 july 2024 evening  july 6th evening 7:30pm")
# af.update_slots("17 july 2024 evening  july 6th evening 7:30pm")
# af.update_slots("july 6th evening 7:30pm")
# print(af.slots)
# print(af.generate_response())