from flask import Flask, render_template, request, redirect
import json
import pandas as pd


def load_state(user_id: int):
    with open(f'state{user_id}.json', 'r') as f:
        state = json.load(f)
    return state


def load_part_metadata(part_data_path):
    with open(part_data_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def persist(state, path):
    with open(path, 'w') as f:
        json.dump(state, f, indent=4)


def parse_example(data_path, sample_index):
    # trying to be memory efficient
    sample = pd.read_csv(data_path, skiprows=lambda x: x not in [0, sample_index])
    he_cont = sample["chat_cut"][0].split('\n')
    return he_cont, sample["gens"][0]


def add_user(user_pids, part_data_path="data/part_data.json"):
    part_metadata = load_part_metadata(part_data_path) 
    if part_metadata["blocks_taken"] < part_metadata["total_blocks"]:
        part_metadata["blocks_taken"] += 1
        user_id = part_metadata["blocks_taken"]
        part_metadata["part_id"][user_pids] = user_id
        persist(part_metadata, part_data_path) 
    else:
        user_id = None

    return user_id

# get experiment data for a user from prolific browser data
def get_user_id(part_data_path='part_data.json'):
    p_pid = request.args.get("PROLIFIC_PID", default=-1, type=str)
    study_id = request.args.get("STUDY_ID", default=-1, type=str)
    session_id = request.args.get("SESSION_ID", default=-1, type=str)
    
    part_metadata = load_part_metadata(part_data_path) 
    user_key = part_metadata["compl_code"] 
    user_pids = str((p_pid, study_id, session_id))
    print(user_pids)
    user_id = part_metadata["part_id"].get(user_pids, None)
    def_user_id = request.args.get("u", default=-1, type=int)
   
    if user_id is None and p_pid != -1 and study_id != -1 and session_id != -1:
        user_id = add_user(user_pids)

    elif def_user_id >= 1 or user_id is None:
        user_id = def_user_id
    
    return user_id, user_key 


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_id, compl_key = get_user_id()
    state_file = f'data/batch_metadata/state{user_id}.json'
    state = load_part_metadata(state_file) 
    he_cont, he_gen = state["default_conv"].values()
    
    if state["page_n"] == 0: 
        if request.method == 'POST':
            state["show_intro"] = False
            he_cont, he_gen = state["solved_example"].values()
        html_page = "intro.html" if state["show_intro"] else "solved_example.html"

    elif state["page_n"] > 15:
        html_page = "outro.html"
        compl_key = compl_key if state["page_n"] == 16 else "" 
    
    else:
        he_cont, he_gen = parse_example(state["he_data_path"], state["page_n"])
        html_page = "index.html"
 
    if request.method == 'POST':
        slider_values = [request.form.get(f'slider{i}') for i in range(1, 5)]
        checkbox_values = request.form.get('checkbox1')
        checkbox_values = checkbox_values == ""
        slider_values.append(checkbox_values)
        print(f"Slider values submitted: {slider_values}")
        if state["page_n"] > 0:
            state["slider_vals"][state["page_n"]] = slider_values 
        state["page_n"] += 1
     
    persist(state, state_file)
    return render_template(
        html_page, 
        he_cont=he_cont,
        he_gen=he_gen,
        header_1=state["header_1"].format(state["page_n"] - 1),
        spec_data=compl_key,
        )
        

if __name__ == '__main__':
    app.run(debug=True)

