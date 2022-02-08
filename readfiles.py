import pandas as pd

texts = []

messages = pd.read_csv('Datasets/Messages.csv')
scores = pd.read_csv('Datasets/Scores.csv')

# Split messages file into its 2 columns
file_names = messages['File Name'].tolist()
file_text = messages['Text'].tolist()

for _ , row in scores.iterrows():
    try:
        curr_message = []
        file_id = row['Merged']
        index = file_names.index(file_id)
        if index:
            # Index text and add to current list
            message = file_text[index]
            curr_message.append(message)

            # Find scores for current row
            env_a = row['env_a']
            comma_a = row['comma_a']
            emp_a = row['emp_a']
            div_a = row['div_a']
            con_a = row['con_a']
            cg_a = row['cg_a']
            hr_a = row['hr_a']
            env_s = row['env_s']
            comm_s = row['comm_s']
            emp_s = row['emp_s']
            div_s = row['div_s']
            con_s = row['con_s']
            cg_s = row['cg_s']
            hr_s = row['hr_s']
            csr = row['csr']

            # Add scores to list of current scores
            curr_message.append(env_a)
            curr_message.append(comma_a)
            curr_message.append(emp_a)
            curr_message.append(div_a)
            curr_message.append(con_a)
            curr_message.append(cg_a)
            curr_message.append(hr_a)
            curr_message.append(env_s)
            curr_message.append(comm_s)
            curr_message.append(emp_s)
            curr_message.append(div_s)
            curr_message.append(con_s)
            curr_message.append(cg_s)
            curr_message.append(hr_s)
            curr_message.append(csr)

            # Add back to current list
            texts.append(curr_message)
    except:
        pass
