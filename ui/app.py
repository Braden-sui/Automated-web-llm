import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

# Configuration for the FastAPI backend
API_BASE_URL = "http://localhost:8000"

# --- Helper Functions to Interact with API ---
def get_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API (Health Check): {e}")
        return None

def list_profiles():
    try:
        response = requests.get(f"{API_BASE_URL}/profiles")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching profiles: {e}")
        return []

# --- Streamlit App Layout ---
st.set_page_config(page_title="Web Automation Control Panel", layout="wide")

st.title("🌐 Web Automation Control Panel")

# Check API Health on load
health_status = get_health()
if health_status and health_status.get("status") == "OK":
    st.sidebar.success(f"API Status: {health_status.get('message')} ({health_status.get('task_runner_status')})")
else:
    st.sidebar.error("API Status: Not Connected or Error")
    st.error("Could not connect to the backend API. Please ensure it is running.")
    st.stop() # Stop rendering the rest of the app if API is down

# --- Main App Sections (using tabs) ---
tab_run, tab_status, tab_profiles = st.tabs(["🚀 Run Task", "📊 Task Status & Results", "📚 Profiles"])

with tab_profiles:
    st.header("Available Task Profiles")
    profiles = list_profiles()
    if profiles:
        profile_df = pd.DataFrame(profiles)
        st.dataframe(profile_df, use_container_width=True)
    else:
        st.info("No profiles found or error fetching profiles.")


# --- "Run Task" Tab --- #
with tab_run:
    st.header("Submit a New Automation Task")

    run_method = st.radio("Choose Task Submission Method:", ("From Profile", "Ad-hoc Instructions"), horizontal=True)

    if run_method == "From Profile":
        st.subheader("Run Task from Profile")
        profiles = list_profiles()
        if not profiles:
            st.warning("No profiles available to run. Please check the API or add profiles.")
        else:
            profile_names = [p['name'] for p in profiles]
            selected_profile_name = st.selectbox("Select Profile:", profile_names)

            st.markdown("**Optional Configuration Overrides (JSON format):**")
            col1, col2, col3 = st.columns(3)
            with col1:
                memory_override_json = st.text_area("Memory Config Override", "{}", height=100, help='e.g., {\"enabled\": true, \"max_items\": 50}')
            with col2:
                visual_override_json = st.text_area("Visual Config Override", "{}", height=100, help='e.g., {\"enabled\": false}')
            with col3:
                browser_override_json = st.text_area("Browser Config Override", "{}", height=100, help='e.g., {\"headless\": false, \"user_agent\": \"custom_agent\"}')

            if st.button("🚀 Run Profile Task", type="primary"):
                if selected_profile_name:
                    try:
                        mem_override = json.loads(memory_override_json) if memory_override_json.strip() and memory_override_json.strip() != "{}" else None
                        vis_override = json.loads(visual_override_json) if visual_override_json.strip() and visual_override_json.strip() != "{}" else None
                        bro_override = json.loads(browser_override_json) if browser_override_json.strip() and browser_override_json.strip() != "{}" else None
                        
                        payload = {}
                        if mem_override: payload['memory_config_override'] = mem_override
                        if vis_override: payload['visual_config_override'] = vis_override
                        if bro_override: payload['browser_config_override'] = bro_override

                        response = requests.post(f"{API_BASE_URL}/tasks/run-profile/{selected_profile_name}", json=payload if payload else None)
                        response.raise_for_status()
                        task_submission_info = response.json()
                        st.success(f"Task '{selected_profile_name}' submitted successfully!")
                        st.json(task_submission_info)
                        st.info("Track its status in the 'Task Status & Results' tab.")
                        # Store submitted task_id in session state to track it later
                        if 'submitted_tasks' not in st.session_state:
                            st.session_state.submitted_tasks = {}
                        st.session_state.submitted_tasks[task_submission_info['task_id']] = {'profile_name': selected_profile_name, 'submitted_at': datetime.now().isoformat(), 'status': 'SUBMITTED'}

                    except json.JSONDecodeError as je:
                        st.error(f"Invalid JSON in configuration overrides: {je}")
                    except requests.exceptions.HTTPError as he:
                        error_detail = he.response.json().get('detail', he.response.text) if he.response else str(he)
                        st.error(f"API Error submitting task: {error_detail}")
                    except requests.exceptions.RequestException as re:
                        st.error(f"Error connecting to API: {re}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
    
    elif run_method == "Ad-hoc Instructions":
        st.subheader("Run Task with Ad-hoc Instructions")
        
        instructions_json = st.text_area("Browser Instructions (JSON list format)", "[]", height=200, help='e.g., [{"type": "NAVIGATE", "url": "https://example.com"}, {"type": "SCREENSHOT", "filename": "example.png"}]')
        
        adhoc_profile_name = st.text_input("Task Name (Optional)", "ad_hoc_ui_task")
        adhoc_description = st.text_input("Task Description (Optional)", "Ad-hoc task submitted via UI.")

        st.markdown("**Optional Configuration Overrides (JSON format):**")
        col_adhoc1, col_adhoc2, col_adhoc3 = st.columns(3)
        with col_adhoc1:
            adhoc_memory_override_json = st.text_area("Ad-hoc Memory Config", "{}", height=100, key="adhoc_mem")
        with col_adhoc2:
            adhoc_visual_override_json = st.text_area("Ad-hoc Visual Config", "{}", height=100, key="adhoc_vis")
        with col_adhoc3:
            adhoc_browser_override_json = st.text_area("Ad-hoc Browser Config", "{}", height=100, key="adhoc_bro")

        if st.button("🚀 Run Ad-hoc Task", type="primary"):
            try:
                instructions = json.loads(instructions_json)
                if not isinstance(instructions, list):
                    st.error("Instructions must be a valid JSON list.")
                    st.stop()

                mem_override = json.loads(adhoc_memory_override_json) if adhoc_memory_override_json.strip() and adhoc_memory_override_json.strip() != "{}" else None
                vis_override = json.loads(adhoc_visual_override_json) if adhoc_visual_override_json.strip() and adhoc_visual_override_json.strip() != "{}" else None
                bro_override = json.loads(adhoc_browser_override_json) if adhoc_browser_override_json.strip() and adhoc_browser_override_json.strip() != "{}" else None

                payload = {
                    "instructions": instructions,
                    "profile_name": adhoc_profile_name or "ad_hoc_ui_task",
                    "description": adhoc_description or "Ad-hoc task submitted via UI.",
                    "memory_config": mem_override,
                    "visual_config": vis_override,
                    "browser_config": bro_override
                }
                # Filter out None config values for a cleaner payload to the API
                payload_cleaned = {k: v for k, v in payload.items() if not (k.endswith('_config') and v is None)}

                response = requests.post(f"{API_BASE_URL}/tasks/run-instructions", json=payload_cleaned)
                response.raise_for_status()
                task_submission_info = response.json()
                st.success(f"Ad-hoc task '{payload_cleaned['profile_name']}' submitted successfully!")
                st.json(task_submission_info)
                st.info("Track its status in the 'Task Status & Results' tab.")
                
                if 'submitted_tasks' not in st.session_state:
                    st.session_state.submitted_tasks = {}
                st.session_state.submitted_tasks[task_submission_info['task_id']] = {'profile_name': payload_cleaned['profile_name'], 'submitted_at': datetime.now().isoformat(), 'status': 'SUBMITTED'}

            except json.JSONDecodeError as je:
                st.error(f"Invalid JSON in instructions or configuration overrides: {je}")
            except requests.exceptions.HTTPError as he:
                error_detail = he.response.json().get('detail', he.response.text) if he.response else str(he)
                st.error(f"API Error submitting ad-hoc task: {error_detail}")
            except requests.exceptions.RequestException as re:
                st.error(f"Error connecting to API: {re}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


# --- Helper Functions for Task Status Tab ---
def get_active_tasks():
    try:
        response = requests.get(f"{API_BASE_URL}/tasks/active")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching active tasks: {e}")
        return {}

def get_task_status(task_id: str):
    try:
        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as he:
        if he.response.status_code == 404:
            st.warning(f"Task ID '{task_id}' not found for status check.")
            return None
        st.error(f"API Error fetching status for task {task_id}: {he.response.json().get('detail', he.response.text)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API for task status: {e}")
        return None

def get_task_result(task_id: str):
    try:
        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}/result")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as he:
        if he.response.status_code == 404:
            st.warning(f"Result for task ID '{task_id}' not found. It might still be running or failed without a full result structure.")
            return None
        st.error(f"API Error fetching result for task {task_id}: {he.response.json().get('detail', he.response.text)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API for task result: {e}")
        return None

def cancel_task_ui(task_id: str):
    try:
        response = requests.post(f"{API_BASE_URL}/tasks/{task_id}/cancel")
        response.raise_for_status()
        st.success(f"Cancellation request sent for task {task_id}. Check status to confirm.")
        return response.json()
    except requests.exceptions.HTTPError as he:
        st.error(f"API Error cancelling task {task_id}: {he.response.json().get('detail', he.response.text)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API for task cancellation: {e}")
        return None

# --- "Task Status & Results" Tab --- #
with tab_status:
    st.header("Monitor Tasks and Retrieve Results")

    # Display Active Tasks
    st.subheader("Active Tasks")
    active_tasks_data = get_active_tasks()
    if active_tasks_data:
        # Convert to DataFrame for better display
        active_tasks_list = [{'task_id': tid, **tinfo} for tid, tinfo in active_tasks_data.items()]
        active_df = pd.DataFrame(active_tasks_list)
        # Reorder columns for clarity
        cols_order = ['task_id', 'profile_name', 'description', 'status', 'submitted_at']
        active_df = active_df[[col for col in cols_order if col in active_df.columns]]
        st.dataframe(active_df, use_container_width=True, hide_index=True)
    else:
        st.info("No active tasks found.")

    st.divider()

    # Query Specific Task
    st.subheader("Query Specific Task by ID")
    task_id_input = st.text_input("Enter Task ID:", help="You can find Task IDs from the 'Active Tasks' list or from submission responses.")

    if task_id_input:
        col_status, col_result, col_cancel = st.columns(3)
        with col_status:
            if st.button("🔍 Get Status", key=f"status_{task_id_input}"):
                status_info = get_task_status(task_id_input)
                if status_info:
                    st.write("**Current Status:**")
                    st.json(status_info)
        
        with col_result:
            if st.button("📄 Get Result", key=f"result_{task_id_input}"):
                result_info = get_task_result(task_id_input)
                if result_info:
                    st.write("**Task Result:**")
                    st.json(result_info) # TaskResult model is complex, JSON is a good way to show it all
        
        with col_cancel:
            if st.button("🛑 Cancel Task", key=f"cancel_{task_id_input}", help="Attempt to cancel a running or pending task."):
                cancel_task_ui(task_id_input)

    st.divider()
    st.subheader("Recently Submitted Tasks (from this session)")
    if 'submitted_tasks' in st.session_state and st.session_state.submitted_tasks:
        recent_df = pd.DataFrame.from_dict(st.session_state.submitted_tasks, orient='index')
        recent_df = recent_df.reset_index().rename(columns={'index': 'task_id'})
        cols_order_recent = ['task_id', 'profile_name', 'status', 'submitted_at']
        recent_df = recent_df[[col for col in cols_order_recent if col in recent_df.columns]]
        st.dataframe(recent_df.sort_values(by='submitted_at', ascending=False), use_container_width=True, hide_index=True)
        
        if st.button("Clear Recent Tasks List (UI only)"):
            st.session_state.submitted_tasks = {}
            st.rerun()
    else:
        st.info("No tasks submitted in this UI session yet.")


# Placeholder for other tabs and functionalities

if __name__ == '__main__':
    # This block is not strictly necessary for Streamlit apps run with `streamlit run`
    # but can be useful if you ever want to run this script directly for some reason (though not typical for Streamlit)
    pass
