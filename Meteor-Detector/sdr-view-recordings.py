import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta
import matplotlib.dates as mdates
import sys # Import sys for graceful exit

class SDRRecordingViewerApp:
    def __init__(self, master):
        self.master = master
        master.title("SDR Recording Viewer")
        master.geometry("1200x800") # Set initial window size

        self.recordings_dir = "recordings"
        self.recordings_data = [] # Stores (display_text, meta_filepath, data_filepath)

        # --- GUI Layout ---
        self.control_frame = ttk.Frame(master, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.plot_frame = ttk.Frame(master)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Controls ---
        ttk.Label(self.control_frame, text="Select Recording:").pack(side=tk.LEFT, padx=5, pady=5)

        self.recording_selector = ttk.Combobox(self.control_frame, state="readonly", width=60)
        self.recording_selector.pack(side=tk.LEFT, padx=5, pady=5)
        self.recording_selector.bind("<<ComboboxSelected>>", self.on_record_selected)

        self.refresh_button = ttk.Button(self.control_frame, text="Refresh List", command=self.load_recordings_list)
        self.refresh_button.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Matplotlib Plot ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.set_title("SDR Magnitude (dB) over Time (UTC)")
        self.ax.set_xlabel("Time (UTC)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.grid(True, linestyle=':', alpha=0.7)

        # Configure UTC time axis formatting
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.fig.autofmt_xdate()

        # Bind mouse wheel events for zooming
        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom) # For Windows/Linux
        # For macOS, you might need to bind to Button-4 and Button-5 events
        # self.canvas.mpl_connect("button_press_event", self.on_scroll_zoom_mac) 

        # Bind the window close button to a custom handler for graceful exit
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load recordings list on startup
        self.load_recordings_list()

    def on_closing(self):
        """Handles the window closing event for graceful exit."""
        print("[GUI] Window closed by user. Exiting gracefully.")
        self.master.quit() # Stop the Tkinter mainloop
        self.master.destroy() # Destroy the Tkinter window
        sys.exit(0) # Ensure the script exits

    def load_recordings_list(self):
        """Scans the recordings directory and populates the dropdown."""
        self.recordings_data = []
        self.recording_selector['values'] = []
        
        if not os.path.exists(self.recordings_dir):
            messagebox.showinfo("No Recordings", f"The directory '{self.recordings_dir}' does not exist yet. Please record some data first.")
            return

        meta_files = [f for f in os.listdir(self.recordings_dir) if f.endswith(".sigmf-meta")]
        
        if not meta_files:
            messagebox.showinfo("No Recordings", f"No SigMF metadata files found in '{self.recordings_dir}'.")
            return

        display_texts = []
        for meta_file in sorted(meta_files): # Sort for consistent order
            meta_filepath = os.path.join(self.recordings_dir, meta_file)
            data_filepath = meta_filepath.replace(".sigmf-meta", ".sigmf-data")

            if not os.path.exists(data_filepath):
                print(f"Warning: Corresponding data file not found for {meta_file}. Skipping.")
                continue

            try:
                with open(meta_filepath, 'r') as f:
                    metadata = json.load(f)
                
                start_time_str = metadata['global']['start_time']
                effective_sample_rate = metadata['global']['sample_rate']
                freq_center = metadata['global']['core:freq_center'] / 1e6 # Convert to MHz
                
                # Parse the ISO 8601 UTC string
                # Handle potential milliseconds and 'Z' timezone indicator
                if start_time_str.endswith('Z'):
                    start_time_str = start_time_str[:-1] # Remove 'Z' for datetime.fromisoformat
                start_datetime_utc = datetime.fromisoformat(start_time_str).replace(tzinfo=timezone.utc)

                # Updated display_text to include full date and time with frequency
                display_text = (f"{os.path.basename(meta_file).replace('.sigmf-meta', '')} "
                                f"(Freq: {freq_center:.2f} MHz, "
                                f"Start: {start_datetime_utc.strftime('%Y-%m-%d %H:%M:%S UTC')})")
                
                self.recordings_data.append((display_text, meta_filepath, data_filepath))
                display_texts.append(display_text)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing metadata from {meta_file}: {e}. Skipping.")
                continue
            
        self.recording_selector['values'] = display_texts
        if display_texts:
            self.recording_selector.set(display_texts[0]) # Select the first item by default
            self.on_record_selected(None) # Trigger plot for the first item

    def on_record_selected(self, event):
        """Callback when a recording is selected from the dropdown."""
        selected_index = self.recording_selector.current()
        if selected_index == -1: # No item selected
            return

        display_text, meta_filepath, data_filepath = self.recordings_data[selected_index]
        self.load_and_plot_data(meta_filepath, data_filepath)

    def load_and_plot_data(self, meta_filepath, data_filepath):
        """Loads data from selected files and updates the plot."""
        try:
            with open(meta_filepath, 'r') as f:
                metadata = json.load(f)

            start_time_str = metadata['global']['start_time']
            effective_sample_rate = metadata['global']['sample_rate']
            sample_count = metadata['captures'][0]['core:sample_count']

            # Parse the ISO 8601 UTC string
            if start_time_str.endswith('Z'):
                start_time_str = start_time_str[:-1]
            start_datetime_utc = datetime.fromisoformat(start_time_str).replace(tzinfo=timezone.utc)

            # Load binary amplitude data (float32)
            # Use np.fromfile for efficiency
            amplitude_data = np.fromfile(data_filepath, dtype=np.float32)

            if len(amplitude_data) != sample_count:
                print(f"Warning: Sample count mismatch in metadata ({sample_count}) vs data file ({len(amplitude_data)}).")
                # Use actual data length for plotting
                sample_count = len(amplitude_data)

            # Reconstruct UTC time axis for plotting
            time_offsets = np.arange(sample_count) / effective_sample_rate
            # Convert time offsets to timedelta objects and add to start_datetime_utc
            # Using vectorized addition for efficiency
            time_axis_utc = np.array([start_datetime_utc + timedelta(seconds=float(s)) for s in time_offsets])

            # Clear previous plot
            self.ax.clear()
            
            # Plot new data
            self.ax.plot(time_axis_utc, amplitude_data, lw=1, color='blue')
            
            self.ax.set_title(f"Magnitude (dB) at {metadata['global']['core:freq_center']/1e6:.2f} MHz\n"
                              f"Start: {start_datetime_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            self.ax.set_xlabel("Time (UTC)")
            self.ax.set_ylabel("Magnitude (dB)")
            self.ax.grid(True, linestyle=':', alpha=0.7)

            # Adjust Y-axis limits
            if len(amplitude_data) > 0:
                finite_data = amplitude_data[np.isfinite(amplitude_data)]
                if len(finite_data) > 0:
                    min_val = np.min(finite_data)
                    max_val = np.max(finite_data)
                    self.ax.set_ylim(min_val - 5, max_val + 5)
                else:
                    self.ax.set_ylim(-100, 0) # Default dB range
            else:
                self.ax.set_ylim(-100, 0) # Default dB range if no data

            # Re-apply date formatting
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.fig.autofmt_xdate()

            self.canvas.draw_idle() # Redraw the canvas

        except Exception as e:
            messagebox.showerror("Error Loading Data", f"Could not load recording:\n{e}")
            print(f"Error loading data: {e}")

    def on_scroll_zoom(self, event):
        """Handles mouse wheel scrolling for zooming."""
        cur_xlim = self.ax.get_xlim()
        xdata = event.xdata  # Get event x location in data coordinates
        
        if xdata is None: # If mouse is outside plot area, do nothing
            return

        zoom_factor = 1.1  # Zoom in/out factor
        if event.button == 'up' or event.button == 'scroll_up': # Zoom in
            new_xlim_min = xdata - (xdata - cur_xlim[0]) / zoom_factor
            new_xlim_max = xdata + (cur_xlim[1] - xdata) / zoom_factor
        elif event.button == 'down' or event.button == 'scroll_down': # Zoom out
            new_xlim_min = xdata - (xdata - cur_xlim[0]) * zoom_factor
            new_xlim_max = xdata + (cur_xlim[1] - xdata) * zoom_factor
        else:
            return

        # Apply new limits
        self.ax.set_xlim(new_xlim_min, new_xlim_max)
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = SDRRecordingViewerApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        # This block catches Ctrl+C from the terminal
        print("\n[GUI] Application interrupted by user (Ctrl+C). Exiting gracefully.")
        root.destroy() # Ensure Tkinter window is destroyed
        sys.exit(0) # Exit the script cleanly

