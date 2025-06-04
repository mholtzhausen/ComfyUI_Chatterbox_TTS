import os
import re
from typing import List, Dict

# Adjusted relative import for AudiobookSection
from .modules.audiobook_models import AudiobookSection

class MH_AudiobookProcessor:
    """
    ComfyUI node for processing text files into sections for an audiobook.

    This class takes a text input with special tags for narrator and character settings,
    parses the text to extract narrator details, character settings, and dialogue,
    splits the text into sections based on a maximum word count per section,
    and creates AudiobookSection objects for each section.

    The text input format supports:
    - Narrator settings at the beginning of the text
    - Character settings using <set> tags
    - Dialogue using character name tags
    - Narrative text interspersed between tags
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the ComfyUI node.

        Returns:
            dict: A dictionary defining the required inputs for the node.
        """
        return {
            "required": {
                "audio_book_text": ("STRING", {
                    "multiline": True,
                    "default": """<narrator voice="joe" cfg=0.4 exp=0.5 tmp=0.8 />
<set name=sunny voice=sarah cfg=0.4 exp=0.7 tmp=0.8 />
<set name="kobus" voice=chris cfg=0.6 exp=0.7 tmp=0.8 />
<set name="janine" voice=ally cfg=0.3 exp=0.8 tmp=0.8 />

The Fantastic Four and the Lost Puppy.

The autumn leaves crunched beneath their feet as Sunny, Koo-bis, Janine, and Saa-rul walked home from school. The afternoon sun painted the sky in beautiful shades of orange and gold, and the four friends were chatting about their exciting day at school when suddenly, they heard a soft whimper coming from behind a large oak tree.

<sunny cfg=0.4 exp=0.7 temp=0.8>"Did you hear that?"</sunny> Sunny asked, pushing her glasses up her nose.

Koo-bis, who was carrying his favorite white guitar with its distinctive red border, stopped strumming mid-chord. <kobus exp=0.6>"It sounds like something's crying."</kobus>

Carefully, they approached the tree, and there, curled up in a pile of golden leaves, was a small brown and white puppy. It was shivering and looked scared, with no collar or tag in sight.

<janine cfg=0.3 exp=0.8>"Oh, the poor thing!"</janine> Janine exclaimed, kneeling down slowly. <janine cfg=0.3 exp=0.8>"It must be lost."</janine>"""
                }),
                "max_words_per_section": ("INT", {
                    "default": 250,
                    "min": 50,
                    "max": 1000,
                    "step": 10
                }),
            }
        }

    RETURN_TYPES = ("AUDIOBOOK_SECTIONS",)
    RETURN_NAMES = ("sections",)
    FUNCTION = "process_audiobook"
    OUTPUT_NODE = False
    CATEGORY = "MH/Chatterbox TTS"

    def process_audiobook(self, audio_book_text: str, max_words_per_section: int) -> tuple[List[AudiobookSection]]:
        """
        Process the audiobook text into sections.

        This method parses the input text to extract narrator details, character settings,
        and dialogue, then splits the text into sections based on the maximum word count
        per section, and creates AudiobookSection objects for each section.

        Args:
            audio_book_text (str): The input text containing narrator settings, character settings,
                                   and dialogue with special tags.
            max_words_per_section (int): The maximum number of words per section.

        Returns:
            tuple[List[AudiobookSection]]: A tuple containing a list of AudiobookSection objects.
        """
        sections: List[AudiobookSection] = []
        narrator_details: Dict = {}
        character_settings: Dict = {}
        current_section_index = 0

        content = audio_book_text
        if not isinstance(audio_book_text, str) or not audio_book_text.strip():
            print(f"Warning: MH_AudiobookProcessor - audio_book_text is empty or invalid.")
            return ([],)

        attribute_parser_regex = re.compile(r"(\w+)=[\"']?([^\"']+)[\"']?")

        def parse_attrs(attr_string: str) -> Dict:
            """
            Parse attribute string into a dictionary.

            Args:
                attr_string (str): The attribute string to parse.

            Returns:
                Dict: A dictionary of parsed attributes.
            """
            attrs = {}
            if attr_string:
                for match in attribute_parser_regex.finditer(attr_string):
                    key, value = match.groups()
                    # Basic type conversion
                    if value.lower() == 'true':
                        attrs[key] = True
                    elif value.lower() == 'false':
                        attrs[key] = False
                    else:
                        try:
                            if '.' in value:
                                attrs[key] = float(value)
                            else:
                                attrs[key] = int(value)
                        except ValueError:
                            attrs[key] = value
            return attrs

        speakable_segments = []

        # Regex for tags
        narrator_tag_regex = re.compile(r"<narrator\s+(.+?)\s*/>", re.IGNORECASE)
        set_tag_regex = re.compile(r"<set\s+name=[\"']?(\w+)[\"']?\s*(.*?)\s*/>", re.IGNORECASE)
        dialogue_tag_regex = re.compile(r"<(\w+)(?:\s+([^>]*?))?\s*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)

        current_pos = 0

        # 1. Parse Narrator Tag (must be at the beginning)
        narrator_match = narrator_tag_regex.match(content)
        if narrator_match:
            narrator_attrs_str = narrator_match.group(1)
            narrator_details.update(parse_attrs(narrator_attrs_str))
            current_pos = narrator_match.end()
        else:
            # Default narrator if not specified
            narrator_details = {"voice_preset": "default_narrator"}
            print("Warning: Narrator tag not found or not at the beginning. Using default narrator settings.")

        # 2. Find all other tags and interspersing text
        all_found_tags = []
        for match in set_tag_regex.finditer(content, current_pos):
            all_found_tags.append({'type': 'set', 'match': match, 'start': match.start()})
        for match in dialogue_tag_regex.finditer(content, current_pos):
            all_found_tags.append({'type': 'dialogue', 'match': match, 'start': match.start()})

        all_found_tags.sort(key=lambda x: x['start'])

        last_processed_pos = current_pos
        for tag_info in all_found_tags:
            match_obj = tag_info['match']
            # Add narrated text before the current tag
            if match_obj.start() > last_processed_pos:
                text_before_tag = content[last_processed_pos:match_obj.start()].strip()
                if text_before_tag:
                    speakable_segments.append({
                        "type": "narrator",
                        "params": narrator_details.copy(),
                        "text": text_before_tag
                    })

            if tag_info['type'] == 'set':
                char_name = match_obj.group(1)
                attrs_str = match_obj.group(2)
                char_attrs = parse_attrs(attrs_str)
                if char_name not in character_settings:
                    character_settings[char_name] = {}
                character_settings[char_name].update(char_attrs)

            elif tag_info['type'] == 'dialogue':
                char_name = match_obj.group(1)
                attrs_str = match_obj.group(2)
                dialogue_text = match_obj.group(3).strip().replace('"', '')

                effective_char_params = character_settings.get(char_name, {}).copy()
                if attrs_str:
                    dialogue_overrides = parse_attrs(attrs_str)
                    effective_char_params.update(dialogue_overrides)

                if dialogue_text:
                    speakable_segments.append({
                        "type": "character",
                        "name": char_name,
                        "params": effective_char_params,
                        "text": dialogue_text
                    })
            last_processed_pos = match_obj.end()

        # Add any remaining narrated text after the last tag
        if last_processed_pos < len(content):
            remaining_text = content[last_processed_pos:].strip()
            if remaining_text:
                speakable_segments.append({
                    "type": "narrator",
                    "params": narrator_details.copy(),
                    "text": remaining_text
                })

        # 3. Split speakable_segments into AudiobookSections
        current_section_text_parts = []
        current_word_count = 0

        for segment in speakable_segments:
            segment_text_clean = segment["text"]
            words_in_segment = len(segment_text_clean.split())

            if not segment_text_clean:
                continue

            # If current section + new segment > max_words, finalize current section
            if current_section_text_parts and (current_word_count + words_in_segment > max_words_per_section):
                sections.append(AudiobookSection(
                    text=" ".join(current_section_text_parts),
                    narrator_details=narrator_details.copy(),
                    characters_details={k: v.copy() for k, v in character_settings.items()},
                    index=current_section_index,
                    filename=f"section_{current_section_index:03d}.wav"
                ))
                current_section_index += 1
                current_section_text_parts = [segment_text_clean]
                current_word_count = words_in_segment
            else:
                # Add segment to current section
                current_section_text_parts.append(segment_text_clean)
                current_word_count += words_in_segment

        # Add any remaining text as the last section
        if current_section_text_parts:
            sections.append(AudiobookSection(
                text=" ".join(current_section_text_parts),
                narrator_details=narrator_details.copy(),
                characters_details={k: v.copy() for k, v in character_settings.items()},
                index=current_section_index,
                filename=f"section_{current_section_index:03d}.wav"
            ))

        return (sections,)