#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import speech_recognition as sr

from blessed import Terminal
from texttable import Texttable

from spacy.matcher import Matcher
import en_core_web_sm
from inference import *
import json 

nlp = en_core_web_sm.load()
nlp.Defaults.stop_words -= {'give', 'Give', 'put', 'Put',
                            'this', 'This', 'that', 'That',
                            'here', 'Here', 'there', 'There'}


matcher = Matcher(nlp.vocab)
term = Terminal()

# r = sr.Recognizer()
# mic = sr.Microphone(device_index=9)
# with mic as source:
#     audio = r.listen(source)
# instruction = r.recognize_google(audio)

# TODO: Add docstring
# TODO: Add language pattern.


class VerbalInstruction:
    """
    Class for the verbal instruction node. It records the verbal
    instruction, transcribes and publishes it.
    """

    def __init__(self, device_index: int, mode='both', print_=True):
        """
        Initializes the instance.

        Parameters
        ----------
        device_index : int, optional
            Index for the audion input that the speech recognizer
            would listen to
        """
        self.speech_recognizer = sr.Recognizer()
        # self.mic = sr.Microphone(device_index=device_index)
        self.instruction_list = []
        self.counter = 1
        self.mode = mode
        self.print = print_
        self.params = {}

        rospy.init_node("verbal_instruction_pub_node", anonymous=False)
        self.verbal_instruction_pub = rospy.Publisher(
            "/verbal_instruction", String, queue_size=10
        )
        self.instruction_msg = ""
        # Run the publisher after initiation
        self.count = 0
        self.commands = [ "put the small red towels in the basket", "Give me the red onions", "Go left", "Grab the large green box on your right", "Put the jar on the table"]
        self.run()

    def run(self):
        """
        Runs the publisher node. Publishes verbal instructions
        received from the user.
        """
        # print('\n\n\n\n\n\n\n\n\n\n\n\n\n')
        rate = rospy.Rate(1)  # 1hz
        while not rospy.is_shutdown():
            # with self.mic as source:
            #     audio = self.speech_recognizer.listen(source)
            # # if no instruction is received, go to the
            # # next iteration.
            # try:
            #     self.instruction_msg = self.speech_recognizer.recognize_google(audio)
            #     if self.instruction_msg in ["exit", "quit"]:
            #         rospy.signal_shutdown("Exiting")
            # except sr.UnknownValueError:
            #     rospy.loginfo("Speaker is quiet")
            #     continue
            # print(term.on_dodgerblue4(f'{term.springgreen}Verbal command: {term.deepskyblue}"{self.instruction_msg}" {term.normal}'))
            if self.count < 5:
                self.instruction_msg = self.commands[self.count]
                self.count += 1
            # if self.count == 5:
            #     rospy.signal_shutdown("Exiting")
            
            
            if self.mode == 'pattern':
                self.params = self.extract_object_info(self.instruction_msg)
            elif self.mode == 'nn':
                self.params = predict(model, self.instruction_msg)
                self.params['no'] = self.counter - 1
            elif self.mode == 'both':
                self.params['pattern'] = self.extract_object_info(self.instruction_msg)
                self.params['nn'] = predict(model, self.instruction_msg)
                self.params['nn']['no'] = self.counter
            
            self.counter += 1
            if self.print:
                print(term.on_dodgerblue4(f'{term.springgreen}Verbal command: {term.deepskyblue}"{self.instruction_msg}" {term.normal}'))
                if self.mode == 'both':
                    print(self.params['pattern'])
                    print(self.params['nn'])
                else:
                    print(self.params)

            
            self.verbal_instruction_pub.publish(json.dumps(self.params))
            rate.sleep()

    def extract_object_info(self, instruction_msg, visualize=False):
        matcher.add("action", [[{"POS": "VERB"},{"POS": "PRON", "OP": "*"},{},{"POS": "ADJ", "OP": "*"},{"POS": "NOUN"}]])
        matcher.add("navigation", [[{"LEMMA": {"IN": ["go", "come", "move", "turn"]}}]])
        matcher.add("attr", [[{"TAG": "JJ", "OP": "+"}, {"POS": "NOUN"}]])
        matcher.add("pos", [[{"LEMMA": {"IN": ["right", "left", "front", "back"]}}]])

        doc = nlp(instruction_msg)
        matches = matcher(doc)
        object_info = {}
        object_name = "None"
        action = "None"
        attr = []
        pos = "None"
        navigation = "None"
        is_navigation = False
        
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            # print(string_id)
            if string_id == "action":
                object_name = doc[end-1].text
                action = doc[start].text
            if string_id == "navigation":
                is_navigation = True
                navigation = doc[start].text
            if string_id == "attr":
                attr.append(doc[start].text)

            if string_id == "pos":
                pos = doc[start].text

        object_info["no"] = self.counter
        object_info["object"] = object_name
        object_info["action"] = navigation if is_navigation else action
        object_info["attr"] = "None" if len(attr) == 0 else attr
        object_info["pos"] = pos
        
        if visualize:
            if ('exit' and 'instruction') not in instruction_msg:
                # print(f'{term.cyan3}Object: {term.purple3}{object_info["object"]} \n{term.cyan3}Action: {term.purple3}{object_info["action"]} \n{term.cyan3}Attributes: {term.purple3}{object_info["attr"]} \n{term.cyan3}Position: {term.purple3}{object_info["pos"]} \n')
                print(f'{term.cyan3}Object: {term.purple3}{object_info["object"]} {term.maroon2}| {term.cyan3}Action: {term.purple3}{object_info["action"]} {term.maroon2}| {term.cyan3}Attributes: {term.purple3}{object_info["attr"]} {term.maroon2}| {term.cyan3}Position: {term.purple3}{object_info["pos"]} \n')
                self.instruction_list.append(object_info)
            elif 'instruction' in instruction_msg:
                table = Texttable()
                instructions_params = [ list(item.values()) for i, item in enumerate(self.instruction_list)]
                table.set_cols_align(["l", "l", "l", "l", "l"])
                table.set_cols_valign(["m", "m","m", "m", "m"])
                table.add_rows([["NO", "Object", "Action", "Attributes", "Position"], *instructions_params])
                print(f'{term.purple3}{table.draw()}')
                print()
            
        return object_info

if __name__ == "__main__":
    VerbalInstruction(device_index=1, mode='both')

# Give me the plate
# Bring me that red cup
# Go left
# Grab the large green box on your right
# Put the jar on the table