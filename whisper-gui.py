import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import whisper
import torch
import numpy as np
import time
import sys
import language_tool_python
import webrtcvad
import tkinter.font as tkfont
import sounddevice as sd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LiveTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Transcription App")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Initialize Whisper model
        self.model_size = "base"
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(self.model_size, device=self.device_type)

        # Initialize LanguageTool
        self.language = "en"
        self.tool = language_tool_python.LanguageTool('en-US')

        # Initialize variables
        self.is_transcribing = False
        self.audio_queue = queue.Queue()
        self.gui_queue = queue.Queue()  # Queue for transcribed text to update the GUI safely
        self.transcribed_text = ""
        self.selected_device = None
        self.model_size_var = tk.StringVar(value=self.model_size)
        self.audio_stream = None

        # Supported languages mapping
        self.languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
        }

        # Whisper model sizes
        self.model_sizes = ["tiny", "base", "small", "medium", "large"]

        # Initialize font settings
        self.transcription_font_family = "Helvetica"
        self.transcription_font_size = 14
        self.transcription_font = (self.transcription_font_family, self.transcription_font_size)

        # Setup UI
        self.setup_ui()

        # Load audio devices
        self.load_devices()

        # Start polling the GUI queue for transcription updates
        self.root.after(100, self.poll_gui_queue)

    def setup_ui(self):
        # Set up style using ttkthemes if available
        try:
            from ttkthemes import ThemedStyle
            self.style = ThemedStyle(self.root)
            self.available_themes = self.style.theme_names()
            self.current_theme = 'arc'
            self.style.set_theme(self.current_theme)
            self.theme_support = True
        except ImportError:
            self.style = ttk.Style(self.root)
            self.available_themes = self.style.theme_names()
            self.current_theme = self.style.theme_use()
            self.style.theme_use(self.current_theme)
            self.theme_support = False

        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Font Settings", command=self.open_font_settings)
        settings_menu.add_command(label="Theme Settings", command=self.open_theme_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        # Layout frames
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, sticky="ew")
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=2, column=0, sticky="ew")

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        middle_frame.grid_rowconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(0, weight=1)

        # Top frame widgets
        # Device selection
        device_label = ttk.Label(top_frame, text="Device:")
        device_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.device_var = tk.StringVar()
        self.device_menu = ttk.Combobox(top_frame, textvariable=self.device_var, state="readonly", width=40)
        self.device_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.device_menu.bind("<<ComboboxSelected>>", self.on_device_change)

        # Language selection
        language_label = ttk.Label(top_frame, text="Language:")
        language_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.language_var = tk.StringVar(value="English")
        self.language_menu = ttk.Combobox(top_frame, textvariable=self.language_var, state="readonly", width=15)
        self.language_menu['values'] = list(self.languages.keys())
        self.language_menu.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.language_menu.bind("<<ComboboxSelected>>", self.on_language_change)

        # Model size selection
        model_label = ttk.Label(top_frame, text="Model Size:")
        model_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.model_menu = ttk.Combobox(top_frame, textvariable=self.model_size_var, state="readonly", width=10)
        self.model_menu['values'] = self.model_sizes
        self.model_menu.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        self.model_menu.bind("<<ComboboxSelected>>", self.on_model_change)

        # Middle frame: Transcription display
        self.transcription_text = tk.Text(middle_frame, wrap=tk.WORD, state=tk.DISABLED, font=self.transcription_font)
        self.transcription_text.grid(row=0, column=0, sticky="nsew")
        self.transcription_scrollbar = ttk.Scrollbar(middle_frame, orient=tk.VERTICAL, command=self.transcription_text.yview)
        self.transcription_scrollbar.grid(row=0, column=1, sticky="ns")
        self.transcription_text.config(yscrollcommand=self.transcription_scrollbar.set)

        # Bottom frame: Control buttons
        self.start_button = ttk.Button(bottom_frame, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.pause_button = ttk.Button(bottom_frame, text="Pause", command=self.pause_transcription, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_button = ttk.Button(bottom_frame, text="Clear Transcription", command=self.clear_transcription)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        # New: Transcribe Audio File button
        self.file_transcribe_button = ttk.Button(bottom_frame, text="Transcribe Audio File", command=self.transcribe_audio_file)
        self.file_transcribe_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = ttk.Button(bottom_frame, text="Save Transcription", command=self.save_transcription, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)


        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_label.grid(row=3, column=0, sticky="ew")

    def load_devices(self):
        try:
            devices = []
            device_info = sd.query_devices()
            unique_devices = set()
            for idx, dev in enumerate(device_info):
                # Add devices with input channels
                if dev['max_input_channels'] > 0:
                    dev_name = f"Input - {dev['name']}"
                    if dev_name not in unique_devices:
                        devices.append((idx, dev_name))
                        unique_devices.add(dev_name)
                # On Windows, attempt to add loopback devices (via WASAPI)
                if sys.platform == 'win32':
                    if dev['hostapi'] is not None:
                        hostapi_info = sd.query_hostapis(dev['hostapi'])
                        if 'WASAPI' in hostapi_info['name'] and dev['max_output_channels'] > 0:
                            loopback_name = f"Output - {dev['name']} (Loopback)"
                            if loopback_name not in unique_devices:
                                devices.append((idx, loopback_name))
                                unique_devices.add(loopback_name)

            self.device_list = devices
            device_names = [name for idx, name in devices]
            self.device_menu['values'] = device_names
            if devices:
                self.device_var.set(device_names[0])
                self.selected_device = devices[0]
            else:
                self.device_var.set('')
                self.selected_device = None
            logging.info(f"Loaded devices: {device_names}")
        except Exception as e:
            logging.error(f"Error loading devices: {e}")
            messagebox.showerror("Error", f"Failed to load audio devices:\n{e}")
            self.device_list = []

    def on_device_change(self, event):
        selected_name = self.device_var.get()
        for dev in self.device_list:
            if dev[1] == selected_name:
                self.selected_device = dev
                logging.info(f"Device changed to: {self.selected_device[1]}")
                self.status_var.set(f"Device changed to: {self.selected_device[1]}")
                break

    def on_language_change(self, event):
        selected_language = self.language_var.get()
        self.language = self.languages.get(selected_language, "en")
        # Update LanguageTool language (adjusting for non-English if needed)
        lang_code = f'{self.language}-US' if self.language == 'en' else self.language
        self.tool = language_tool_python.LanguageTool(lang_code)
        logging.info(f"Language changed to: {self.language}")

    def on_model_change(self, event):
        selected_model = self.model_size_var.get()
        if selected_model != self.model_size:
            self.model_size = selected_model
            threading.Thread(target=self.reload_model, daemon=True).start()

    def reload_model(self):
        try:
            self.status_var.set(f"Loading model '{self.model_size}'...")
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.is_transcribing = False
            if self.audio_stream:
                self.audio_stream.close()
                self.audio_stream = None
            self.model = whisper.load_model(self.model_size, device=self.device_type)
            messagebox.showinfo("Model Loaded", f"Whisper model '{self.model_size}' loaded successfully.")
            self.start_button.config(state=tk.NORMAL)
            self.status_var.set("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model '{self.model_size}': {e}")
            self.model_size_var.set(self.model_size)
            self.status_var.set("Ready")

    def start_transcription(self):
        if not self.selected_device:
            messagebox.showerror("Error", "No device selected.")
            return
        if self.is_transcribing:
            messagebox.showinfo("Info", "Transcription is already running.")
            return

        self.is_transcribing = True
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL, text="Pause")
        self.save_button.config(state=tk.DISABLED)
        self.transcribed_text = ""
        self.transcription_text.config(state=tk.NORMAL)
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.config(state=tk.DISABLED)
        self.status_var.set("Transcription started...")
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()
        # Start transcription worker thread
        self.transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        self.transcription_thread.start()

    def process_audio(self):
        try:
            device_index = self.selected_device[0]
            RATE = 16000  # 16kHz sample rate
            CHANNELS = 1
            FRAME_DURATION_MS = 30
            FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
            vad = webrtcvad.Vad()
            vad.set_mode(3)  # Most aggressive

            self.speech_buffer = []
            self.num_silence_frames = 0
            self.max_silence_frames = int(0.5 / (FRAME_DURATION_MS / 1000))
            self.max_speech_frames = int(30 / (FRAME_DURATION_MS / 1000))

            def callback(indata, frames, time_info, status):
                if not self.is_transcribing:
                    raise sd.CallbackStop()
                audio_data = indata[:, 0]  # Mono channel
                audio_data_int16 = (audio_data * 32768).astype(np.int16)
                frame_bytes = audio_data_int16.tobytes()
                is_speech = vad.is_speech(frame_bytes, sample_rate=RATE)
                if is_speech:
                    self.speech_buffer.append(audio_data_int16)
                    self.num_silence_frames = 0
                else:
                    self.num_silence_frames += 1
                    if self.num_silence_frames > self.max_silence_frames and self.speech_buffer:
                        speech_audio = np.concatenate(self.speech_buffer)
                        speech_audio_float = speech_audio.astype(np.float32) / 32768.0
                        self.audio_queue.put(speech_audio_float)
                        self.speech_buffer = []
                        self.num_silence_frames = 0
                if len(self.speech_buffer) > self.max_speech_frames:
                    speech_audio = np.concatenate(self.speech_buffer)
                    speech_audio_float = speech_audio.astype(np.float32) / 32768.0
                    self.audio_queue.put(speech_audio_float)
                    self.speech_buffer = []
                    self.num_silence_frames = 0

            self.audio_stream = sd.InputStream(samplerate=RATE,
                                               device=device_index,
                                               channels=CHANNELS,
                                               dtype='float32',
                                               callback=callback,
                                               blocksize=FRAME_SIZE)
            self.audio_stream.start()
            logging.info("Audio stream started.")
            while self.is_transcribing:
                time.sleep(0.1)
            # Stop the stream when transcription stops
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
            if self.speech_buffer:
                speech_audio = np.concatenate(self.speech_buffer)
                speech_audio_float = speech_audio.astype(np.float32) / 32768.0
                self.audio_queue.put(speech_audio_float)
                self.speech_buffer = []
            self.status_var.set("Ready")
            logging.info("Audio processing completed.")
        except Exception as e:
            logging.error(f"Error in process_audio: {e}")
            self.is_transcribing = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.status_var.set("Ready")

    def transcription_worker(self):
        while self.is_transcribing or not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get(timeout=1)
                # Validate that we have a 1-D NumPy array
                if not isinstance(audio_data, np.ndarray) or audio_data.ndim != 1:
                    logging.error("Invalid audio data received.")
                    continue
                options = {
                    "language": self.language,
                    "task": "transcribe",
                    "fp16": torch.cuda.is_available(),
                    "temperature": 0,
                }
                result = self.model.transcribe(audio_data, **options)
                segment_text = result.get('text', '').strip()
                corrected_text = self.tool.correct(segment_text)
                # Pass the corrected text to the GUI update queue
                self.gui_queue.put(corrected_text + " ")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in transcription_worker: {e}")
                self.gui_queue.put(f"\n[Error in transcription: {e}]\n")
                self.is_transcribing = False
                self.start_button.config(state=tk.NORMAL)
                self.pause_button.config(state=tk.DISABLED)
                self.status_var.set("Ready")
                break

    def poll_gui_queue(self):
        try:
            while not self.gui_queue.empty():
                text_segment = self.gui_queue.get_nowait()
                self.transcribed_text += text_segment
                self.transcription_text.config(state=tk.NORMAL)
                self.transcription_text.delete(1.0, tk.END)
                self.transcription_text.insert(tk.END, self.transcribed_text)
                self.transcription_text.see(tk.END)
                self.transcription_text.config(state=tk.DISABLED)
        except Exception as e:
            logging.error(f"Error updating GUI: {e}")
        finally:
            self.root.after(100, self.poll_gui_queue)

    def pause_transcription(self):
        if self.is_transcribing:
            self.is_transcribing = False
            self.pause_button.config(text="Resume")
            self.save_button.config(state=tk.NORMAL)
            self.status_var.set("Transcription paused.")
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                logging.info("Audio stream paused.")
        else:
            if not self.selected_device:
                messagebox.showerror("Error", "No device selected.")
                return
            self.is_transcribing = True
            self.pause_button.config(text="Pause")
            self.save_button.config(state=tk.DISABLED)
            self.status_var.set("Transcription resumed.")
            self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.audio_thread.start()
            self.transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
            self.transcription_thread.start()

    def clear_transcription(self):
        self.transcribed_text = ""
        self.transcription_text.config(state=tk.NORMAL)
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.config(state=tk.DISABLED)
        self.status_var.set("Transcription cleared.")

    def save_transcription(self):
        if not self.transcribed_text.strip():
            messagebox.showwarning("No Text", "There is no transcribed text to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.transcribed_text)
                messagebox.showinfo("Success", f"Transcription saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save transcription: {e}")

    def open_font_settings(self):
        font_dialog = tk.Toplevel(self.root)
        font_dialog.title("Font Settings")
        font_dialog.geometry("300x200")
        font_dialog.resizable(False, False)

        font_family_label = ttk.Label(font_dialog, text="Font Family:")
        font_family_label.pack(pady=5)

        font_families = sorted(list(tkfont.families()))
        self.font_family_var = tk.StringVar(value=self.transcription_font_family)
        font_family_menu = ttk.Combobox(font_dialog, textvariable=self.font_family_var, values=font_families, state="readonly")
        font_family_menu.pack(pady=5)

        font_size_label = ttk.Label(font_dialog, text="Font Size:")
        font_size_label.pack(pady=5)

        self.font_size_var = tk.IntVar(value=self.transcription_font_size)
        font_size_spinbox = ttk.Spinbox(font_dialog, from_=8, to=72, textvariable=self.font_size_var)
        font_size_spinbox.pack(pady=5)

        apply_button = ttk.Button(font_dialog, text="Apply", command=lambda: [self.apply_font_settings(), font_dialog.destroy()])
        apply_button.pack(pady=10)

    def apply_font_settings(self):
        self.transcription_font_family = self.font_family_var.get()
        self.transcription_font_size = self.font_size_var.get()
        self.transcription_font = (self.transcription_font_family, self.transcription_font_size)
        self.transcription_text.config(font=self.transcription_font)
        self.status_var.set("Font settings applied.")

    def open_theme_settings(self):
        theme_dialog = tk.Toplevel(self.root)
        theme_dialog.title("Theme Settings")
        theme_dialog.geometry("300x150")
        theme_dialog.resizable(False, False)

        theme_label = ttk.Label(theme_dialog, text="Select Theme:")
        theme_label.pack(pady=5)

        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_menu = ttk.Combobox(theme_dialog, textvariable=self.theme_var, values=self.available_themes, state="readonly")
        theme_menu.pack(pady=5)

        apply_button = ttk.Button(theme_dialog, text="Apply", command=lambda: [self.apply_theme_settings(), theme_dialog.destroy()])
        apply_button.pack(pady=10)

    def apply_theme_settings(self):
        selected_theme = self.theme_var.get()
        if self.theme_support:
            self.style.set_theme(selected_theme)
        else:
            self.style.theme_use(selected_theme)
        self.current_theme = selected_theme
        self.status_var.set(f"Theme changed to '{selected_theme}'.")

    def show_about(self):
        messagebox.showinfo("About", "Live Transcription App\nVersion 1.0\nDeveloped by Brady Meighan")

    def on_closing(self):
        self.is_transcribing = False
        # Allow threads time to finish
        time.sleep(1)
        if self.audio_stream:
            self.audio_stream.close()
        self.root.destroy()

    def transcribe_audio_file(self):
        """
        Opens a file dialog for the user to select an audio file and transcribes it using the Whisper model.
        """
        # (Optional) Prevent concurrent processing if live transcription is running.
        if self.is_transcribing:
            messagebox.showwarning("Warning", "Please pause live transcription before transcribing an audio file.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")]
        )
        if file_path:
            # Clear any existing transcription and update status.
            self.clear_transcription()
            self.status_var.set("Transcribing audio file...")
            # Start transcription in a new thread to avoid freezing the GUI.
            threading.Thread(target=self.process_file_transcription, args=(file_path,), daemon=True).start()

    def process_file_transcription(self, file_path):
        """
        Processes the transcription of the selected audio file.
        """
        try:
            options = {
                "language": self.language,
                "task": "transcribe",
                "fp16": torch.cuda.is_available(),
                "temperature": 0,
            }
            # Transcribe the file directly (Whisper accepts a filepath)
            result = self.model.transcribe(file_path, **options)
            segment_text = result.get('text', '').strip()
            corrected_text = self.tool.correct(segment_text)
            # Push the corrected transcription into the GUI update queue
            self.gui_queue.put(corrected_text)
            # Update the status (using root.after to ensure thread-safe GUI update)
            self.root.after(0, lambda: self.status_var.set("File transcription complete."))
        except Exception as e:
            logging.error(f"Error transcribing file: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error transcribing file:\n{e}"))
            self.root.after(0, lambda: self.status_var.set("Ready"))


def main():
    try:
        root = tk.Tk()
        app = LiveTranscriptionApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        logging.error(f"Exception in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
