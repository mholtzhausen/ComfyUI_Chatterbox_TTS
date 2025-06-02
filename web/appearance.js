import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "MH_Nodes.appearance",
  async nodeCreated(node) {
    if (node.comfyClass.startsWith("MH_")) {
      // Apply styling
      node.color = "#865A04";
      node.bgcolor = "#4F0074";
    }
  },
});