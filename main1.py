import os
import tempfile
import time
import threading
from datetime import datetime
import pandas as pd
from PIL import Image
import streamlit as st
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sqlite3
import json
from utils import detect_and_process_id_card

# Database setup
def init_database():
    """Initialize SQLite database for storing ID card data"""
    conn = sqlite3.connect('id_cards_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS id_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            first_name TEXT,
            last_name TEXT,
            full_name TEXT,
            national_id TEXT UNIQUE,
            address TEXT,
            birth_date TEXT,
            governorate TEXT,
            gender TEXT,
            processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence_score REAL,
            validation_status TEXT DEFAULT 'pending'
        )
    ''')
    
    conn.commit()
    conn.close()

def save_to_database(data, filename, confidence_score=0.0):
    """Save extracted data to database"""
    conn = sqlite3.connect('id_cards_database.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO id_cards 
            (filename, first_name, last_name, full_name, national_id, address, 
             birth_date, governorate, gender, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filename, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], confidence_score))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.warning(f"National ID {data[3]} already exists in database")
        return False
    except Exception as e:
        st.error(f"Database error: {e}")
        return False
    finally:
        conn.close()

def export_to_excel():
    """Export database to Excel file"""
    conn = sqlite3.connect('id_cards_database.db')
    df = pd.read_sql_query("SELECT * FROM id_cards", conn)
    conn.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"id_cards_export_{timestamp}.xlsx"
    df.to_excel(filename, index=False)
    return filename

def get_database_stats():
    """Get database statistics"""
    conn = sqlite3.connect('id_cards_database.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM id_cards")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM id_cards WHERE validation_status = 'validated'")
    validated_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT governorate, COUNT(*) FROM id_cards GROUP BY governorate")
    governorate_stats = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_records': total_records,
        'validated_records': validated_records,
        'governorate_stats': governorate_stats
    }

# File monitoring system
class IDCardHandler(FileSystemEventHandler):
    def __init__(self, process_callback):
        self.process_callback = process_callback
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in self.supported_formats:
                # Wait a moment to ensure file is fully written
                time.sleep(2)
                self.process_callback(file_path)

def start_folder_monitoring(folder_path, process_callback):
    """Start monitoring a folder for new images"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    event_handler = IDCardHandler(process_callback)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    return observer

def process_multiple_images(image_files, progress_bar=None):
    """Process multiple images and return results"""
    results = []
    total_files = len(image_files)
    
    for i, image_file in enumerate(image_files):
        try:
            if hasattr(image_file, 'read'):  # Streamlit uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(image_file.read())
                    temp_file_path = temp_file.name
                filename = image_file.name
            else:  # File path
                temp_file_path = image_file
                filename = os.path.basename(image_file)
            
            # Process the image
            data = detect_and_process_id_card(temp_file_path)
            
            # Calculate confidence score (placeholder - you can implement actual confidence calculation)
            confidence_score = 0.85  # This should be calculated based on OCR confidence
            
            # Save to database
            save_to_database(data, filename, confidence_score)
            
            results.append({
                'filename': filename,
                'data': data,
                'status': 'success',
                'confidence': confidence_score
            })
            
            # Clean up temporary file if created
            if hasattr(image_file, 'read'):
                os.remove(temp_file_path)
                
        except Exception as e:
            results.append({
                'filename': filename if 'filename' in locals() else 'unknown',
                'data': None,
                'status': 'error',
                'error': str(e)
            })
        
        # Update progress bar
        if progress_bar:
            progress_bar.progress((i + 1) / total_files)
    
    return results

# Streamlit configuration
st.set_page_config(
    page_title='Enhanced Egyptian ID Card Processor',
    page_icon='💳',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize database
init_database()

# Initialize session state
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False
if "observer" not in st.session_state:
    st.session_state.observer = None

# Sidebar navigation
st.sidebar.title("🏛️ Egyptian ID Processor")
tabs = ["Home", "Batch Processing", "Auto Monitor", "Database", "Analytics", "Settings", "Guide"]
selected_tab = st.sidebar.radio("Navigation", tabs)
st.session_state.current_tab = selected_tab

# Home Tab - Single Image Processing
if st.session_state.current_tab == "Home":
    st.title("Egyptian ID Card Processor 💳")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an ID card image",
            type=['webp', 'jpg', 'tif', 'tiff', 'png', 'mpo', 'bmp', 'jpeg', 'dng', 'pfm']
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if not uploaded_file:
            st.image("ocr2.png", use_container_width=True)
        else:
            with st.spinner("Processing ID card..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name
                    
                    data = detect_and_process_id_card(temp_file_path)
                    
                    # Save to database
                    save_to_database(data, uploaded_file.name, 0.85)
                    
                    # Display results
                    st.success("✅ Processing completed!")
                    
                    if os.path.exists("d2.jpg"):
                        st.image("d2.jpg", caption="Detected Fields", use_container_width=True)
                    
                    st.markdown("### Extracted Information:")
                    
                    info_cols = st.columns(2)
                    with info_cols[0]:
                        st.metric("First Name", data[0])
                        st.metric("Last Name", data[1])
                        st.metric("National ID", data[3])
                        st.metric("Birth Date", data[5])
                    
                    with info_cols[1]:
                        st.metric("Full Name", data[2])
                        st.metric("Governorate", data[6])
                        st.metric("Gender", data[7])
                        st.text_area("Address", data[4], height=100)
                
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
                finally:
                    if 'temp_file_path' in locals():
                        os.remove(temp_file_path)

# Batch Processing Tab
elif st.session_state.current_tab == "Batch Processing":
    st.title("Batch Processing 📁")
    
    st.info("Process multiple ID card images at once")
    
    uploaded_files = st.file_uploader(
        "Upload multiple ID card images",
        type=['webp', 'jpg', 'tif', 'tiff', 'png', 'mpo', 'bmp', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files for processing")
        
        if st.button("🚀 Process All Images", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing images..."):
                results = process_multiple_images(uploaded_files, progress_bar)
            
            # Display results summary
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = len(results) - successful
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(results))
            with col2:
                st.metric("Successful", successful, delta=successful)
            with col3:
                st.metric("Failed", failed, delta=-failed if failed > 0 else 0)
            
            # Detailed results
            if st.checkbox("Show detailed results"):
                for result in results:
                    with st.expander(f"📄 {result['filename']} - {result['status'].upper()}"):
                        if result['status'] == 'success':
                            data = result['data']
                            st.write(f"**Name:** {data[2]}")
                            st.write(f"**National ID:** {data[3]}")
                            st.write(f"**Birth Date:** {data[5]}")
                            st.write(f"**Governorate:** {data[6]}")
                            st.write(f"**Confidence:** {result['confidence']:.2%}")
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")

# Auto Monitor Tab
elif st.session_state.current_tab == "Auto Monitor":
    st.title("Automatic Folder Monitoring 👁️")
    
    st.info("Monitor a folder for new ID card images and process them automatically")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        monitor_folder = st.text_input(
            "Folder Path to Monitor",
            value=f"D:\\Bonyanex\\auto monitor",
            help="Enter the path to the folder you want to monitor"
        )
    
    with col2:
        st.write("Status:")
        if st.session_state.monitoring_active:
            st.success("🟢 Monitoring Active")
        else:
            st.error("🔴 Monitoring Inactive")
    
    def process_new_file(file_path):
        """Callback function for processing new files"""
        try:
            data = detect_and_process_id_card(file_path)
            filename = os.path.basename(file_path)
            save_to_database(data, filename, 0.80)
            st.success(f"✅ Processed: {filename}")
        except Exception as e:
            st.error(f"❌ Failed to process {file_path}: {e}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Start Monitoring", disabled=st.session_state.monitoring_active):
            try:
                observer = start_folder_monitoring(monitor_folder, process_new_file)
                st.session_state.observer = observer
                st.session_state.monitoring_active = True
                st.success(f"Started monitoring: {monitor_folder}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start monitoring: {e}")
    
    with col2:
        if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.monitoring_active):
            if st.session_state.observer:
                st.session_state.observer.stop()
                st.session_state.observer.join()
                st.session_state.observer = None
                st.session_state.monitoring_active = False
                st.success("Monitoring stopped")
                st.rerun()
    
    if st.session_state.monitoring_active:
        st.markdown("### 📋 Monitoring Instructions:")
        st.markdown(f"""
        1. Drop ID card images into the folder: `{monitor_folder}`
        2. The system will automatically detect and process new images
        3. Results will be saved to the database automatically
        4. Check the Database tab to view processed records
        """)

# Database Tab
elif st.session_state.current_tab == "Database":
    st.title("Database Management 🗄️")
    
    # Database statistics
    stats = get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", stats['total_records'])
    with col2:
        st.metric("Validated Records", stats['validated_records'])
    with col3:
        validation_rate = (stats['validated_records'] / max(stats['total_records'], 1)) * 100
        st.metric("Validation Rate", f"{validation_rate:.1f}%")
    
    # Export functionality
    st.markdown("### 📤 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to Excel"):
            filename = export_to_excel()
            st.success(f"✅ Data exported to: {filename}")
            
            # Provide download link
            with open(filename, "rb") as file:
                st.download_button(
                    label="📥 Download Excel File",
                    data=file.read(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col2:
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.checkbox("⚠️ I understand this will delete all records"):
                conn = sqlite3.connect('id_cards_database.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM id_cards")
                conn.commit()
                conn.close()
                st.success("Database cleared successfully")
                st.rerun()
    
    # View recent records
    st.markdown("### 📋 Recent Records")
    conn = sqlite3.connect('id_cards_database.db')
    df = pd.read_sql_query(
        "SELECT * FROM id_cards ORDER BY processing_date DESC LIMIT 10", 
        conn
    )
    conn.close()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records found in database")

# Analytics Tab
elif st.session_state.current_tab == "Analytics":
    st.title("Analytics Dashboard 📊")
    
    conn = sqlite3.connect('id_cards_database.db')
    df = pd.read_sql_query("SELECT * FROM id_cards", conn)
    conn.close()
    
    if df.empty:
        st.info("No data available for analytics")
    else:
        # Governorate distribution
        st.markdown("### 🏛️ Distribution by Governorate")
        gov_counts = df['governorate'].value_counts()
        st.bar_chart(gov_counts)
        
        # Gender distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👥 Gender Distribution")
            gender_counts = df['gender'].value_counts()
            st.pie_chart(gender_counts)
        
        with col2:
            st.markdown("### 📅 Processing Timeline")
            df['processing_date'] = pd.to_datetime(df['processing_date'])
            daily_counts = df.groupby(df['processing_date'].dt.date).size()
            st.line_chart(daily_counts)
        
        # Age distribution (if birth dates are available)
        if 'birth_date' in df.columns and not df['birth_date'].isna().all():
            st.markdown("### 🎂 Age Distribution")
            df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
            current_year = datetime.now().year
            df['age'] = current_year - df['birth_date'].dt.year
            age_groups = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], labels=['<18', '18-30', '31-45', '46-60', '60+'])
            age_counts = age_groups.value_counts()
            st.bar_chart(age_counts)

# Settings Tab
elif st.session_state.current_tab == "Settings":
    st.title("Settings ⚙️")
    
    st.markdown("### 🔧 Processing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### OCR Settings")
        ocr_confidence = st.slider("OCR Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
        enable_preprocessing = st.checkbox("Enable Image Preprocessing", True)
        auto_validation = st.checkbox("Auto Validation", False)
    
    with col2:
        st.markdown("#### Database Settings")
        auto_backup = st.checkbox("Auto Backup", True)
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
        max_records = st.number_input("Max Records to Keep", 1000, 100000, 10000)
    
    st.markdown("### 📁 File Handling")
    
    col1, col2 = st.columns(2)
    with col1:
        supported_formats = st.multiselect(
            "Supported Image Formats",
            ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'],
            default=['jpg', 'jpeg', 'png']
        )
    
    with col2:
        delete_processed = st.checkbox("Delete Processed Images", False)
        archive_processed = st.checkbox("Archive Processed Images", True)
    
    if st.button("💾 Save Settings"):
        settings = {
            'ocr_confidence': ocr_confidence,
            'enable_preprocessing': enable_preprocessing,
            'auto_validation': auto_validation,
            'auto_backup': auto_backup,
            'backup_frequency': backup_frequency,
            'max_records': max_records,
            'supported_formats': supported_formats,
            'delete_processed': delete_processed,
            'archive_processed': archive_processed
        }
        
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        
        st.success("✅ Settings saved successfully!")

# Guide Tab
elif st.session_state.current_tab == "Guide":
    st.title("User Guide 📖")
    
    st.markdown("""
    ## 🚀 Enhanced Egyptian ID Card Processing System
    
    ### Features Overview:
    
    #### 🏠 **Home - Single Image Processing**
    - Upload and process individual ID card images
    - Real-time extraction of personal information
    - Automatic database storage
    - Visual field detection display
    
    #### 📁 **Batch Processing**
    - Process multiple images simultaneously
    - Progress tracking and detailed results
    - Bulk database insertion
    - Error handling and reporting
    
    #### 👁️ **Auto Monitor**
    - Automatic folder monitoring for new images
    - Real-time processing of detected files
    - Background operation with status indicators
    - Configurable folder paths
    
    #### 🗄️ **Database Management**
    - SQLite database for reliable storage
    - Excel export functionality
    - Record validation and management
    - Data cleanup and maintenance tools
    
    #### 📊 **Analytics Dashboard**
    - Statistical analysis of processed data
    - Governorate and gender distributions
    - Processing timeline visualization
    - Age group analysis
    
    #### ⚙️ **Settings**
    - Configurable OCR parameters
    - File handling preferences
    - Backup and archiving options
    - System optimization settings
    
    ### 📋 **How to Use:**
    
    1. **Single Processing**: Upload an image in the Home tab
    2. **Batch Processing**: Select multiple images in Batch Processing tab
    3. **Auto Monitoring**: Set up folder monitoring in Auto Monitor tab
    4. **View Data**: Check Database tab for all processed records
    5. **Analytics**: Explore data insights in Analytics tab
    6. **Configuration**: Adjust settings in Settings tab
    
    ### 🔧 **Technical Requirements:**
    
    - Python 3.8+
    - Required packages: streamlit, pandas, pillow, watchdog, sqlite3
    - YOLO models: detect_id_card.pt, detect_objects.pt, detect_id.pt
    - EasyOCR with Arabic language support
    
    ### 💡 **Tips for Best Results:**
    
    - Ensure good image quality and lighting
    - Keep ID cards flat and unfolded
    - Avoid shadows and reflections
    - Use high-resolution images when possible
    - Regular database backups recommended
    
    ### 🆘 **Troubleshooting:**
    
    - **OCR Errors**: Check image quality and preprocessing settings
    - **Database Issues**: Verify file permissions and disk space
    - **Monitoring Problems**: Ensure folder exists and is accessible
    - **Performance**: Adjust batch size and confidence thresholds
    """)
    
    st.markdown("---")
    st.info("💬 For support and feature requests, please contact the development team.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🏛️ **Egyptian ID Processor v2.0**")
st.sidebar.markdown("Enhanced with batch processing and monitoring")

# Cleanup on app shutdown
if st.session_state.get('observer'):
    import atexit
    def cleanup():
        if st.session_state.observer:
            st.session_state.observer.stop()
            st.session_state.observer.join()
    
    atexit.register(cleanup)