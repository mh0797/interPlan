from typing import Any


class ModificationsSerializableDictionary:
    list_of_modifications = {
        "a": "amount_of_agents",
        "d": "density",
        "g": "goal",
        "o": "observation",
        "s": "special_scenario",
    }

    density_modification_character_to_command = {
            "l": "low",
            "m": "medium",
            "h": "high",
        }

    def __init__(self, dictionary) -> None:
        self.dictionary = dictionary

    def add_scenario_specifics(self, string):
        """
        Add new entries to the dictionary from "scenario specifics"
        """
        assert isinstance(
            string, str
        ), f"Class to be of type {str}, but is {type(string)}!"

        for idx, letter in enumerate(string):
            if idx % 2 != 0:
                continue  # TODO this doesn't work if amount of agents has more than one digit
            if letter == "a":
                self.dictionary["amount_of_agents"] = int(string[idx + 1])
            else:
                self.dictionary[self.get_name_of_mod(letter)] = string[idx + 1]

    def to_string(self) -> str:
        string = ""
        for letter, name in self.list_of_modifications.items():
            if name in self.dictionary:
                if name == "special_scenario":
                    return f"s{self.dictionary['special_scenario']}"
                else:
                    string += f"{letter + str(self.dictionary[name])}"
        return string

    @staticmethod
    def get_name_of_mod(letter: str) -> str:
        return ModificationsSerializableDictionary.list_of_modifications[letter]

    def reset_scenario_specifics(self):
        for name_of_mod in self.list_of_modifications.values():
            if name_of_mod in self.dictionary:
                self.dictionary.pop(name_of_mod)

    def augment_agents(self) -> bool:
        """
        Returns True if there are modifications done to agents
        """
        return (
            "density" in self.dictionary
            or "amount_of_agents" in self.dictionary
            or "special_scenario" in self.dictionary
        )

    def __call__(self) -> dict:
        return self.dictionary
