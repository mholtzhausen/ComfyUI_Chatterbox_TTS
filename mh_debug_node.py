import json
from comfy.comfy_types.node_typing import IO
from server import PromptServer # Import PromptServer

class MH_Debug:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_value": (IO.ANY, {})
            },
            # Add hidden inputs for the JavaScript to identify the node
            "hidden": {
                "unique_id": "UNIQUE_ID", # ComfyUI populates this with a unique ID for the node
                "extra_pnginfo": "EXTRA_PNGINFO", # Can be used to trigger updates
            },
        }

    RETURN_TYPES = () # No standard output wires
    FUNCTION = "main"
    OUTPUT_NODE = True # Still useful for direct execution if no outputs

    CATEGORY = "MH/Utils"

    def main(self, input_value=None, unique_id=None, extra_pnginfo=None):
        value = 'None'
        if isinstance(input_value, str):
            value = input_value
        elif isinstance(input_value, (int, float, bool)):
            value = str(input_value)
        elif input_value is not None:
            try:
                value = json.dumps(input_value, indent=2) # Add indent for readability
            except Exception:
                try:
                    value = str(input_value)
                except Exception:
                    value = 'source exists, but could not be serialized.'

        # Send the value to the frontend using PromptServer
        # This message will be caught by your JavaScript extension
        if unique_id: # Ensure unique_id is available
            # Print to console for easy debugging
            print(f"Debug node {unique_id} value type: {type(input_value).__name__}")
            print(f"Debug node {unique_id} value: {value}")
            PromptServer.instance.send_sync("mh_debug.update_text", {
                "unique_id": unique_id,
                "text": value
            })

        # Do NOT return {"ui": {"text": (value,)}} if you want it *in* the node.
        # This return causes output to the standard "Text Output" panel.
        # Since OUTPUT_NODE is True, we don't need a return for wire outputs.
        return {} # Return an empty dictionary, or just nothing if not an OUTPUT_NODE