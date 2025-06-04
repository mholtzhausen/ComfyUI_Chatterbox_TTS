import { app } from "../../scripts/app.js"; // Import the main ComfyUI app instance

console.log("mh_debug.display: Extension loaded");
app.registerExtension({
  name: "mh_debug.display", // Unique name for your extension
  async setup() {
    // Function to update or create a custom widget in the node
    function updateNodeText(node, text) {
      // Try to find an existing widget for our display
      let displayWidget = node.widgets.find(
        (w) => w.name === "mh_debug_display"
      );

      if (!displayWidget) {
        console.log(
          `mh_debug.display: Creating new widget for node ${node.name}`
        );
        // If the widget doesn't exist, create it
        displayWidget = node.addCustomWidget({
          name: "mh_debug_display", // Unique name for this widget
          draw: function (ctx, node, width, y) {
            // This function draws the widget content
            ctx.fillStyle = "white"; // Text color
            ctx.font = "12px monospace"; // Font style, monospace for code-like output
            const padding = 10;
            let currentY = y + padding; // Start drawing below the input ports

            const lines = this.value.split("\n");
            lines.forEach((line) => {
              ctx.fillText(line, padding, currentY);
              currentY += 15; // Line height
            });
          },
          computeSize: function (width) {
            // Dynamically adjust height based on content
            const lines = this.value.split("\n").length;
            const lineHeight = 15;
            const paddingTop = 10;
            const paddingBottom = 10;
            return [
              width,
              Math.max(30, lines * lineHeight + paddingTop + paddingBottom),
            ];
          },
          // We don't need a default value, it's set by backend
          value: text,
          // Make it non-interactive
          input: false,
        });
        node.setDirtyCanvas(true, true); // Redraw the node
      } else {
        // Update the existing widget's value
        if (displayWidget.value !== text) {
          // Only update if content changed
          displayWidget.value = text;
          node.setDirtyCanvas(true, true); // Redraw the node
        }
      }
    }

    // Listen for the custom event sent from your Python node
    app.api.addEventListener("mh_debug.update_text", (event) => {
      const data = event.detail;
      const unique_id = data.unique_id;
      const text_to_display = data.text;
      console.log(
        `mh_debug.display: Updating node with unique_id ${unique_id} with text: ${text_to_display}`
      );

      // Find the node instance by its unique_id
      for (const node of app.graph.nodes) {
        // The unique_id is stored in node.properties
        if (node.properties && node.properties.unique_id === unique_id) {
          updateNodeText(node, text_to_display);
          break; // Found the node, no need to continue
        }
      }
    });
  },
});
