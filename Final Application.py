import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Global variables
df_global = None
current_fig = None
current_ax = None
recent_files = []

# Default sensor thresholds
sensor_thresholds = {
    'Temp': 45,
    'Vibration': 0.5,
    'Voltage': 250,
    'Proximity': 20,
    'Current': 10,
    'Pressure': 100,
    'Humidity': 70,
    'Load': 15,
}

# ML functions
## =============================== Motor Overheating =============================== ## 

def Motor_Overheating_faults(dataframe):
    df = dataframe.copy()

    # Step 1: Create fault labels using rule-based logic
    df['Fault_Status'] = ((df['Temp'] > 45) & (df['Current'] > 5)).astype(int)

    # Step 2: Select features and labels
    features = ['Temp', 'Current', 'Vibration', 'DC_Motor_Speed']
    X = df[features]
    y = df['Fault_Status']

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict on the entire dataset
    df['ML_Prediction'] = model.predict(X)

    # Step 6: Count number of predicted faults in full data
    fault_count = df['ML_Prediction'].sum()

    # Step 7: Return summary
    if fault_count > 0:
        return f"{fault_count} Faults, Temperature & Current increased"
    else:
        return "Normal"

## =============================== Conveyor Belt Jam =============================== ## 

def detect_conveyor_jam(dataframe):
    df = dataframe.copy()

    # Step 1: Rule-based fault labeling
    df['Fault_Status'] = ((df['Proximity'] < 10) & (df['Load'] > 15) & (df['Current'] > 5)).astype(int)

    # Step 2: Feature selection
    features = ['Proximity', 'Load', 'Current', 'Temp', 'DC_Motor_Speed']
    X = df[features]
    y = df['Fault_Status']

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict on full dataset
    df['ML_Prediction'] = model.predict(X)

    # Step 6: Fault summary
    fault_count = df['ML_Prediction'].sum()

    # Step 7: Return summary
    if fault_count > 0:
        return f"{fault_count} Faults, Conveyor jam detected (obstruction/overweight)"
    else:
        return "Normal"
    
## =============================== Humidity Damage Risk =============================== ##

def detect_humidity_damage(dataframe):
    df = dataframe.copy()

    # Step 1: Rule-based labeling
    df['Fault_Status'] = (df['Humidity'] > 70).astype(int)

    # Step 2: Feature selection
    features = ['Humidity', 'Temp', 'Current', 'Vibration']
    X = df[features]
    y = df['Fault_Status']

    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict
    df['ML_Prediction'] = model.predict(X)

    # Step 6: Count predicted faults
    fault_count = df['ML_Prediction'].sum()

    # Step 7: Return summary
    if fault_count > 0:
        return f"{fault_count} Faults, High humidity detected (risk of damage)"
    return "Normal"

## =============================== AGV Navigation Failure =============================== ##

def detect_agv_navigation_failure(dataframe):
    df = dataframe.copy()

    # Step 1: Rule-based labeling
    df['Fault_Status'] = ((df['Motion_Detected'] == 'No') & (df['Optical_State'] == 'Misaligned')).astype(int)

    # Step 2: Encode categorical columns
    df_encoded = df.copy()
    label_encoders = {}
    for col in ['Motion_Detected', 'Optical_State']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le  # Optional: Save encoders for inverse_transform if needed

    # Step 3: Feature selection
    features = ['Motion_Detected', 'Optical_State', 'DC_Motor_Speed', 'Proximity']
    X = df_encoded[features]
    y = df_encoded['Fault_Status']

    # Step 4: Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict
    df['ML_Prediction'] = model.predict(X)

    # Step 6: Fault count
    fault_count = df['ML_Prediction'].sum()

    # Step 7: Summary
    if fault_count > 0:
        return f"{fault_count} Faults, AGV navigation failure (obstruction or misalignment)"
    return "Normal"

## =============================== Sorting System Error =============================== ##

def detect_sorting_error(dataframe):
    df = dataframe.copy()

    # Step 1: Rule-based labeling
    df['Fault_Status'] = ((df['Optical_State'] == 'Misaligned') & (df['Proximity'] < 10)).astype(int)

    # Step 2: Encode Optical_State
    df_encoded = df.copy()
    le = LabelEncoder()
    df_encoded['Optical_State'] = le.fit_transform(df_encoded['Optical_State'])

    # Step 3: Feature selection
    features = ['Optical_State', 'Proximity', 'DC_Motor_Speed', 'Servo_Angle']
    X = df_encoded[features]
    y = df_encoded['Fault_Status']

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Predict
    df['ML_Prediction'] = model.predict(X)

    # Step 7: Summary
    fault_count = df['ML_Prediction'].sum()
    if fault_count > 0:
        return f"{fault_count} Faults, Sorting system error (misalignment or package misplacement)"
    return "Normal"

## =============================== Power Surge/Drop fault =============================== ##

def detect_power_surge_drop(dataframe):
    df = dataframe.copy()

    # Label creation
    df['Current_Diff'] = df['Current'].diff().abs()
    voltage_issue = (df['Voltage'] < 200) | (df['Voltage'] > 250)
    current_fluctuation = df['Current_Diff'] > 1
    df['Fault_Status'] = (voltage_issue & current_fluctuation).astype(int)

    # Feature set
    df['Voltage_Deviation'] = df['Voltage'] - 230
    features = ['Voltage', 'Voltage_Deviation', 'Current', 'Current_Diff', 'DC_Motor_Speed']
    df.fillna(0, inplace=True)  # handle NaN in first row due to diff
    X = df[features]
    y = df['Fault_Status']

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and summarize
    df['ML_Prediction'] = model.predict(X)
    fault_count = df['ML_Prediction'].sum()

    if fault_count > 0:
        return f"{fault_count} Faults, Power surge/drop predicted (voltage/current instability)"
    return "Normal"

## =============================== Motor Overload fault =============================== ##

def detect_motor_overload(dataframe):
    df = dataframe.copy()

    # Step 1: Label the fault using rule
    df['Fault_Status'] = ((df['Current'] > 5) & (df['Temp'] > 45)).astype(int)

    # Step 2: Features
    features = ['Temp', 'Current', 'Vibration', 'DC_Motor_Speed']
    df.fillna(0, inplace=True)  # handle NaN
    X = df[features]
    y = df['Fault_Status']

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict
    df['ML_Prediction'] = model.predict(X)

    # Step 6: Summary
    fault_count = df['ML_Prediction'].sum()
    if fault_count > 0:
        return f"{fault_count} Faults, Motor overload predicted (high current + temperature > 45°C)"
    return "Normal"

## =============================== Bearing Wear =============================== ##

def detect_bearing_wear(dataframe):
    df = dataframe.copy()

    # Step 1: Label the fault using rule
    df['Bearing_Wear_Status'] = (df['Vibration'] > 0.5).astype(int)

    # Step 2: Features and target
    features = ['Vibration']
    df.fillna(0, inplace=True)  # handle NaN
    X = df[features]
    y = df['Bearing_Wear_Status']

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict
    df['ML_Prediction'] = model.predict(X)

    # Step 6: Set actuator response
    df['Buzzer'] = df['ML_Prediction'].apply(lambda x: 'ON' if x == 1 else 'OFF')

    # Step 7: Summary
    fault_count = df['ML_Prediction'].sum()
    if fault_count > 0:
        return f"{fault_count} Bearing Wear Faults Detected, Vibrations Increased, Abnormal Noise→ Buzzer ON"
    return "Normal"

def show_file_load_window():
    """Show the enhanced file loading window"""
    root = tk.Tk()
    root.title("Warehouse Fault Detector")
    root.geometry("800x600")
    root.configure(bg='#f0f0f0')
    
    # Header Frame
    header_frame = tk.Frame(root, bg='#2c3e50')
    header_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Application Title
    title_label = tk.Label(header_frame, 
                         text="SMART WAREHOUSE PREDICTIVE MAINTENANCE SYSTEM", # WAREHOUSE SENSOR DATA ANALYSIS PLATFORM
                         font=('Helvetica', 20, 'bold'),
                         fg='white', bg='#2c3e50')
    title_label.pack(pady=20)
    
    # Main Content Frame
    content_frame = tk.Frame(root, bg='#f0f0f0')
    content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
    
    # Left Panel - Recent Files
    left_panel = tk.Frame(content_frame, bg='#ecf0f1', bd=2, relief=tk.RAISED)
    left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    
    recent_label = tk.Label(left_panel, 
                          text="RECENT FILES", 
                          font=('Helvetica', 12, 'bold'),
                          bg='#3498db', fg='white')
    recent_label.pack(fill=tk.X, pady=5)
    
    recent_listbox = tk.Listbox(left_panel, 
                               height=15, 
                               width=30,
                               font=('Helvetica', 10),
                               selectbackground='#3498db')
    recent_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Load recent files into listbox
    for file in recent_files[-5:]:  # Show last 5 recent files
        recent_listbox.insert(tk.END, os.path.basename(file))
    
    def load_selected_file():
        """Load file selected from recent files list"""
        selection = recent_listbox.curselection()
        if selection:
            file_path = recent_files[selection[0]]
            try:
                load_and_process_file(root, file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file:\n{str(e)}")
    
    recent_button = ttk.Button(left_panel, 
                             text="Open Selected", 
                             command=load_selected_file)
    recent_button.pack(fill=tk.X, padx=5, pady=5)
    
    # Right Panel - Main Options
    right_panel = tk.Frame(content_frame, bg='#f0f0f0')
    right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
    
    # Welcome Message
    welcome_frame = tk.Frame(right_panel, bg='#f0f0f0')
    welcome_frame.pack(fill=tk.X, pady=20)
    
    welcome_label = tk.Label(welcome_frame, 
                           text="Welcome to Warehouse Sensor Data Analysis Platform", 
                           font=('Helvetica', 14),
                           bg='#f0f0f0')
    welcome_label.pack()
    
    desc_label = tk.Label(welcome_frame, 
                        text="Analyze and visualize sensor data with powerful tools", 
                        font=('Helvetica', 10),
                        bg='#f0f0f0')
    desc_label.pack()
    
    # Main Buttons Frame
    buttons_frame = tk.Frame(right_panel, bg='#f0f0f0')
    buttons_frame.pack(expand=True)
    
    # Load New File Button
    def load_new_file():
        """Handle new file loading"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            load_and_process_file(root, file_path)
    
    load_btn = ttk.Button(buttons_frame, 
                         text="LOAD NEW DATA FILE", 
                         command=load_new_file,
                         style='W.TButton')
    load_btn.pack(pady=15, ipadx=20, ipady=10)
    
    # Quick Start Guide Button
    def show_quick_guide():
        """Show quick start guide"""
        guide = """
        QUICK START GUIDE:
        
        1. LOAD DATA: Click 'Load New Data File' to import your sensor data
        2. SELECT SENSORS: Choose which sensors to analyze from the list
        3. SET THRESHOLDS: Adjust threshold values for each sensor
        4. GENERATE PLOTS: Visualize sensor data with threshold violations
        5. RUN ANALYSIS: Use ML tools to detect potential faults
        
        Supported file formats: Excel (.xlsx, .xls), CSV (.csv)
        Required column: 'Timestamp' (in datetime format)
        """
        messagebox.showinfo("Quick Start Guide", guide)
    
    quick_btn = ttk.Button(buttons_frame, 
                          text="QUICK START GUIDE", 
                          command=show_quick_guide)
    quick_btn.pack(pady=10, ipadx=20, ipady=5)
    
    # Settings Button
    def show_settings():
        """Show application settings"""
        messagebox.showinfo("Settings", "Application settings will be available in the next version")
    
    settings_btn = ttk.Button(buttons_frame, 
                            text="APPLICATION SETTINGS", 
                            command=show_settings)
    settings_btn.pack(pady=10, ipadx=20, ipady=5)
    
    # Footer Frame
    footer_frame = tk.Frame(root, bg='#2c3e50')
    footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
    
    version_label = tk.Label(footer_frame, 
                           text="Sensor Analysis Tool v1.0.1 | © 2025", 
                           font=('Helvetica', 8),
                           fg='white', bg='#2c3e50')
    version_label.pack(pady=5)
    
    # Configure styles
    style = ttk.Style()
    style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'), foreground='white', background='#3498db')
    
    root.mainloop()

def load_and_process_file(root_window, file_path):
    """Load and process the selected file with enhanced timestamp parsing"""
    global df_global, recent_files
    
    try:
        # Read file based on extension
        if file_path.endswith(('.xlsx', '.xls')):
            df_global = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df_global = pd.read_csv(file_path)
        else:
            messagebox.showerror("Error", "Unsupported file format")
            return
        
        if 'Timestamp' not in df_global.columns:
            messagebox.showerror("Error", "No 'Timestamp' column found in the data")
            return

        # Enhanced timestamp parsing with multiple format support
        timestamp_formats = [
            '%Y-%m-%d %H:%M:%S',  # ISO format with seconds
            '%Y-%m-%d %H:%M',     # ISO format without seconds
            '%d-%m-%Y %H:%M:%S',  # European with dashes
            '%d-%m-%Y %H:%M',     # European with dashes (no seconds)
            '%m-%d-%Y %H:%M:%S',  # US format with dashes
            '%m-%d-%Y %H:%M',     # US format with dashes (no seconds)
            '%d/%m/%Y %H:%M:%S',  # European with slashes
            '%d/%m/%Y %H:%M',     # European with slashes (no seconds)
            '%m/%d/%Y %H:%M:%S',  # US with slashes
            '%m/%d/%Y %H:%M',     # US with slashes (no seconds)
            '%Y%m%d %H%M%S',      # Compact format with seconds
            '%Y%m%d %H%M',        # Compact format without seconds
            '%d-%b-%Y %H:%M:%S',  # With month abbreviation
            '%d %b %Y %H:%M:%S',  # With month abbreviation and spaces
            'mixed'               # Pandas auto-detect as last resort
        ]

        # Try parsing with each format
        parsing_success = False
        for fmt in timestamp_formats:
            try:
                if fmt == 'mixed':
                    df_global['Timestamp'] = pd.to_datetime(
                        df_global['Timestamp'], 
                        errors='coerce',
                        infer_datetime_format=True
                    )
                else:
                    df_global['Timestamp'] = pd.to_datetime(
                        df_global['Timestamp'], 
                        format=fmt, 
                        errors='coerce'
                    )
                
                # Check if we successfully parsed all timestamps
                if df_global['Timestamp'].isna().sum() == 0:
                    parsing_success = True
                    break
                    
            except ValueError:
                continue

        # Handle invalid timestamps
        invalid_count = df_global['Timestamp'].isna().sum()
        if invalid_count > 0:
            # Get sample invalid timestamps (first 3)
            invalid_samples = df_global.loc[
                df_global['Timestamp'].isna(), 'Timestamp'
            ].iloc[:3].tolist()
            
            # Show warning with details
            warning_msg = (
                f"Could not parse {invalid_count} timestamp(s).\n"
                f"First few problematic values: {invalid_samples}\n\n"
                "The problematic rows will be dropped."
            )
            
            if not parsing_success:
                warning_msg += (
                    "\n\nNote: Some timestamps were parsed successfully, "
                    "but others failed. This suggests inconsistent formats "
                    "in your data."
                )
            
            messagebox.showwarning("Timestamp Warning", warning_msg)
            
            # Drop invalid rows and reset index
            df_global = df_global.dropna(subset=['Timestamp']).reset_index(drop=True)

        # Update recent files list
        if file_path in recent_files:
            recent_files.remove(file_path)
        recent_files.append(file_path)
        if len(recent_files) > 5:
            recent_files.pop(0)

        root_window.destroy()
        open_main_window()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

def return_to_file_load(main_window):
    """Return to file loading window"""
    if messagebox.askyesno("Confirm", "Return to file selection? Current analysis will be lost."):
        main_window.destroy()
        show_file_load_window()

def generate_plot():
    """Generate the sensor plot with matplotlib zoom tools"""
    global current_fig, current_ax
    
    if df_global is None:
        messagebox.showerror("Error", "No data loaded.")
        return

    selected_sensors = [sensor for sensor, var in sensor_vars.items() if var.get()]
    if not selected_sensors:
        messagebox.showwarning("Warning", "No sensors selected.")
        return

    try:
        start = pd.to_datetime(graph_start_date.get())
        end = pd.to_datetime(graph_end_date.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid date format.")
        return

    df_filtered = df_global[(df_global['Timestamp'] >= start) & (df_global['Timestamp'] <= end)]
    if df_filtered.empty:
        messagebox.showinfo("Info", "No data found for selected date range.")
        return

    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    current_fig, current_ax = plt.subplots(figsize=(8, 5))
    
    # Store threshold violation counts
    violation_counts = {}
    
    for sensor in selected_sensors:
        current_ax.plot(df_filtered['Timestamp'], df_filtered[sensor], label=sensor)
        threshold = float(threshold_entries[sensor].get()) if threshold_entries[sensor].get() else sensor_thresholds.get(sensor, 0)
        
        if sensor != 'Proximity':
            mask = df_filtered[sensor] > threshold
        else:
            mask = df_filtered[sensor] < threshold
        
        violation_counts[sensor] = sum(mask)
        current_ax.plot(df_filtered['Timestamp'][mask], df_filtered[sensor][mask], 
                       'rx', markersize=8, label=f'{sensor} Threshold Violation')

    current_ax.xaxis_date()
    current_fig.autofmt_xdate()
    
    date_format = plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')
    current_ax.xaxis.set_major_formatter(date_format)
    
    time_range = end - start
    if time_range <= pd.Timedelta(hours=6):
        locator = plt.matplotlib.dates.MinuteLocator(interval=30)
    elif time_range <= pd.Timedelta(days=1):
        locator = plt.matplotlib.dates.HourLocator(interval=2)
    else:
        locator = plt.matplotlib.dates.DayLocator()
    current_ax.xaxis.set_major_locator(locator)

    counts_text = "\n".join([f"{sensor}: {count} violations" for sensor, count in violation_counts.items()])
    current_ax.text(0.02, 0.98, counts_text, 
                   transform=current_ax.transAxes,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    current_ax.set_title("Sensor Readings with Threshold Violations")
    current_ax.set_xlabel("Time")
    current_ax.set_ylabel("Value")
    # Add toggle-able legend
    legend = current_ax.legend()
    legend.set_visible(False)  # Start hidden
    
    # Add legend toggle button
    legend_frame = ttk.Frame(plot_frame)
    legend_frame.pack(fill=tk.X, pady=5)
    
    def toggle_legend():
        legend.set_visible(not legend.get_visible())
        current_fig.canvas.draw_idle()
    
    ttk.Button(legend_frame, 
              text="Toggle Legend", 
              command=toggle_legend,
              style='TButton').pack(side=tk.LEFT)
    current_ax.grid(True)

    # Create canvas with navigation toolbar
    canvas_plot = FigureCanvasTkAgg(current_fig, master=plot_frame)
    canvas_plot.draw()
    
    # Add matplotlib navigation toolbar
    toolbar = NavigationToolbar2Tk(canvas_plot, plot_frame)
    toolbar.update()
    
    canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar.pack(fill=tk.X)

def run_ml_fault_detection():
    """Run ML fault detection algorithms"""
    if df_global is None:
        return
    
    faults_text.delete(1.0, tk.END)
    
    try:
        start = pd.to_datetime(fault_start_date.get())
        end = pd.to_datetime(fault_end_date.get())
        df_filtered = df_global[(df_global['Timestamp'] >= start) & (df_global['Timestamp'] <= end)]
    except ValueError:
        messagebox.showerror("Error", "Invalid date format in fault detection range.")
        return
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    faults_text.insert(tk.END, f"=== ML FAULT DETECTION RESULTS (Run at: {current_time}) ===\n\n", 'header')
    faults_text.insert(tk.END, f"Analysis Period: {start} to {end}\n\n", 'subheader')
    
    results = [
        ("🔧 Motor Overheating", Motor_Overheating_faults),
        ("⚙ Bearing Wear", detect_bearing_wear),
        ("🔄 Conveyor Jam", detect_conveyor_jam),
        ("💧 Humidity Damage", detect_humidity_damage),
        ("🤖 AGV Navigation", detect_agv_navigation_failure),
        ("📦 Sorting System", detect_sorting_error),
        ("⚡ Power Surge/Drop", detect_power_surge_drop),
        ("🏋️ Motor Overload", detect_motor_overload)
    ]
    
    for name, func in results:
        faults_text.insert(tk.END, f"{name}:\n", 'subheader')
        try:
            result = func(df_filtered)
            faults_text.insert(tk.END, f"{result}\n\n")
        except Exception as e:
            faults_text.insert(tk.END, f"Error in detection: {str(e)}\n\n", 'error')


def open_main_window():
    """Open the main application window with enhanced UI and detachable sensor controls"""
    global sensor_vars, graph_start_date, graph_end_date, fault_start_date, fault_end_date
    global plot_frame, faults_text, threshold_entries, current_fig, current_ax

    main_window = tk.Tk()
    main_window.title("Sensor Monitoring with ML Fault Prediction")
    main_window.geometry("1200x800")
    
    # ======================
    # Modern Style Configuration
    # ======================
    style = ttk.Style()
    style.theme_use('clam')
    
    # Custom colors
    bg_color = '#f5f6fa'
    accent_color = '#487eb0'
    panel_color = '#dfe6e9'
    warning_color = '#e84118'
    
    style.configure('TFrame', background=bg_color)
    style.configure('TLabel', background=bg_color, font=('Helvetica', 10))
    style.configure('TButton', font=('Helvetica', 10), padding=5)
    style.configure('Accent.TButton', background=accent_color, foreground='white', font=('Helvetica', 10, 'bold'))
    style.configure('Warning.TButton', background=warning_color, foreground='white')
    style.configure('TEntry', padding=5)
    
    main_window.configure(bg=bg_color)
    
    # ======================
    # Header Bar
    # ======================
    header_frame = ttk.Frame(main_window, style='TFrame')
    header_frame.pack(fill=tk.X, pady=5, padx=10)
    
    title_label = ttk.Label(header_frame, 
                          text="WAREHOUSE SENSOR DATA ANALYTICS DASHBOARD", 
                          font=('Helvetica', 16, 'bold'),
                          foreground=accent_color)
    title_label.pack(side=tk.LEFT)
    
    back_button = ttk.Button(header_frame, 
                           text="← Back to File Selection", 
                           command=lambda: return_to_file_load(main_window),
                           style='TButton')
    back_button.pack(side=tk.RIGHT, padx=5)
    
    # ======================
    # Time Range Controls
    # ======================
    time_frame = ttk.Frame(main_window, style='TFrame')
    time_frame.pack(fill=tk.X, pady=10, padx=10)
    
    # Graph time range
    ttk.Label(time_frame, 
             text="VISUALIZATION RANGE:", 
             font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w')
    
    ttk.Label(time_frame, text="From:").grid(row=1, column=0, padx=5)
    graph_start_date = ttk.Entry(time_frame)
    graph_start_date.grid(row=1, column=1)
    graph_start_date.insert(0, df_global['Timestamp'].min().strftime("%Y-%m-%d %H:%M"))
    
    ttk.Label(time_frame, text="To:").grid(row=1, column=2, padx=5)
    graph_end_date = ttk.Entry(time_frame)
    graph_end_date.grid(row=1, column=3)
    graph_end_date.insert(0, df_global['Timestamp'].max().strftime("%Y-%m-%d %H:%M"))
    
    # Fault detection range
    ttk.Label(time_frame, 
             text="FAULT DETECTION RANGE:", 
             font=('Helvetica', 10, 'bold')).grid(row=0, column=6, padx=(20,5))
    
    ttk.Label(time_frame, text="From:").grid(row=1, column=6, padx=5)
    fault_start_date = ttk.Entry(time_frame)
    fault_start_date.grid(row=1, column=7)
    fault_start_date.insert(0, df_global['Timestamp'].min().strftime("%Y-%m-%d %H:%M"))
    
    ttk.Label(time_frame, text="To:").grid(row=1, column=8, padx=5)
    fault_end_date = ttk.Entry(time_frame)
    fault_end_date.grid(row=1, column=9)
    fault_end_date.insert(0, df_global['Timestamp'].max().strftime("%Y-%m-%d %H:%M"))
    
    
    
    # ======================
    # Main Content Area (Modified Widths)
    # ======================
    main_frame = ttk.Frame(main_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # Left Panel - Visualization (80% width)
    left_panel = ttk.Frame(main_frame, width=900)
    left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Right Panel - Fault Detection (20% width)
    right_panel = ttk.Frame(main_frame, width=250)
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
    
    # ======================
    # Sensor Popup Button (Replaces old sensor container)
    # ======================
    def open_sensor_popup():
        """Open sensor controls in a separate window"""
        popup = tk.Toplevel(main_window)
        popup.title("Sensor Selection & Thresholds")
        popup.geometry("400x500")
        popup.update_idletasks()
        x = main_window.winfo_x() + (main_window.winfo_width() - popup.winfo_width()) // 2
        y = main_window.winfo_y() + (main_window.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{x}+{y}")
         # Store thresholds before destruction 
        def on_close():
            # Save all threshold values before closing
            for col in numeric_cols:
                if threshold_entries[col].get():
                    sensor_thresholds[col] = float(threshold_entries[col].get())
            popup.destroy()
        
        popup.protocol("WM_DELETE_WINDOW", on_close)
        
        # Header
        ttk.Label(popup, 
                 text="SELECT SENSORS & SET THRESHOLDS", 
                 font=('Helvetica', 12, 'bold'),
                 foreground=accent_color).pack(pady=10)
        
        # Scrollable frame
        canvas = tk.Canvas(popup, bg='white')
        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add sensor controls
        for col in numeric_cols:
            row_frame = ttk.Frame(scroll_frame)
            row_frame.pack(fill=tk.X, pady=3, padx=10)
            
            ttk.Checkbutton(row_frame, 
                          text=col, 
                          variable=sensor_vars[col]).pack(side=tk.LEFT)
            
            ttk.Label(row_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)
            threshold_entries[col] = ttk.Entry(row_frame, width=8)
            threshold_entries[col].pack(side=tk.LEFT)
            
            if col in sensor_thresholds:
                threshold_entries[col].insert(0, str(sensor_thresholds[col]))
        
        
        # Change the close button command
        ttk.Button(popup, 
                text="Apply & Close", 
                command=on_close,  # Use on_close instead of direct destroy
                style='Accent.TButton').pack(pady=10)
    
    buttons_row = ttk.Frame(left_panel)
    buttons_row.pack(pady=5)

    # First button
    ttk.Button(buttons_row, 
            text="⚙️ Open Sensor Controls", 
            command=open_sensor_popup, 
            style='Accent.TButton').grid(row=0, column=0, padx=5)

    # Second button
    ttk.Button(buttons_row, 
            text="Generate Plot", 
            command=generate_plot, 
            style='Accent.TButton').grid(row=0, column=1, padx=5)
    
    # ======================
    # Plot Area with Legend Toggle
    # ======================
    plot_frame = ttk.Frame(left_panel)
    plot_frame.pack(fill=tk.BOTH, expand=True)
    
    # ======================
    # Fault Detection Panel
    # ======================
    fault_header = ttk.Frame(right_panel)
    fault_header.pack(fill=tk.X, pady=5)
    
    ttk.Label(fault_header, 
             text="FAULT DETECTION RESULTS", 
             font=('Helvetica', 12, 'bold'),
             foreground=accent_color).pack(side=tk.LEFT)
    
    ttk.Button(fault_header, 
              text="Run Analysis", 
              command=run_ml_fault_detection,
              style='Accent.TButton').pack(side=tk.RIGHT)
    
    # Results text area
    faults_text = tk.Text(right_panel, 
                         height=30, 
                         wrap=tk.WORD, 
                         bg='white', 
                         padx=10, 
                         pady=10,
                         font=('Consolas', 9))
    
    text_scroll = ttk.Scrollbar(right_panel, command=faults_text.yview)
    faults_text.configure(yscrollcommand=text_scroll.set)
    
    faults_text.tag_configure('header', font=('Helvetica', 10, 'bold'), foreground=accent_color)
    faults_text.tag_configure('subheader', font=('Helvetica', 9, 'bold'), foreground=accent_color)
    faults_text.tag_configure('error', foreground=warning_color)
    
    faults_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    # ======================
    # Initialize sensor variables
    # ======================
    sensor_vars = {}
    threshold_entries = {}
    numeric_cols = [col for col in df_global.columns if col != 'Timestamp' and pd.api.types.is_numeric_dtype(df_global[col])]
    
    for col in numeric_cols:
        sensor_vars[col] = tk.BooleanVar(value= False)  # Default all sensors selected
    
    main_window.mainloop()

if __name__ == "__main__":
    show_file_load_window()