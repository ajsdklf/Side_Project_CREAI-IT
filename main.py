from openai import OpenAI
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os
from llama_index.core import Prompt

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

st.header("장애아 교육자를 위한 챗봇")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(v1, v2):
    # 두 벡터의 내적(dot product)
    dot_product = np.dot(v1, v2)

    # 각 벡터의 노름(norm)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 코사인 유사도 계산
    similarity = dot_product / (norm_v1 * norm_v2)

    return similarity

disorder_DB = {
    'Autism Spectrum Disorder (ASD)': 'A developmental disorder characterized by difficulties with social interaction, communication, and often includes repetitive behaviors and restricted interests. Individuals with ASD may also display unique strengths and differences in the way they perceive the world.',
    'Attention-Deficit/Hyperactivity Disorder (ADHD)': 'A neurodevelopmental disorder marked by an ongoing pattern of inattention and/or hyperactivity-impulsivity that interferes with functioning or development. ADHD may affect focus, the ability to sit still, and self-control.',
    'Down Syndrome': 'A genetic disorder caused by the presence of all or part of a third copy of chromosome 21. It is associated with physical growth delays, characteristic facial features, and mild to moderate intellectual disability.',
    'Cerebral Palsy': 'A group of disorders that affect a person’s ability to move and maintain balance and posture. Cerebral palsy is the result of brain damage that occurs before or during birth, or during the first few years of life.',
    'Sensory Processing Disorder': 'A condition in which the brain has trouble receiving and responding to information that comes in through the senses. This may manifest in oversensitivity to stimuli in the environment, poor motor skills, or being uncoordinated.'
}


disorder_behavior_DB = {
    'ASD': {
        'misbehavior1': 'Avoids or resists physical contact which might be interpreted as unaffectionate behavior.',
        'misbehavior2': 'Has difficulty in playing interactively with other children, potentially leading to social isolation.',
        'misbehavior3': 'Engages in repetitive movements such as rocking or spinning, which might be disruptive.',
        'misbehavior4': 'May have intense tantrums as a response to sensory overload.',
        'misbehavior5': 'Displays difficulties with transitions resulting in resistance or distress when changing activities.'
    },
    'ADHD': {
        'misbehavior1': 'Interrupts others frequently, which can be perceived as impolite.',
        'misbehavior2': 'Has trouble staying seated or in one place, often seen as restlessness.',
        'misbehavior3': 'Acts without thinking, leading to potential accidents or unsafe situations.',
        'misbehavior4': 'Struggles with following instructions, often not completing tasks.',
        'misbehavior5': 'Exhibits impulsive behaviors that can disrupt group settings.'
    },
    'Down Syndrome': {
        'misbehavior1': 'May exhibit stubbornness or refusal to participate in activities.',
        'misbehavior2': 'Has difficulty with verbal communication which can lead to frustrations and outbursts.',
        'misbehavior3': 'Shows a preference for solitary play which can hinder social development.',
        'misbehavior4': 'Can be overly affectionate in inappropriate settings.',
        'misbehavior5': 'Might ignore personal space boundaries of others.'
    },
    'Cerebral Palsy': {
        'misbehavior1': 'Has involuntary movements which can lead to knocking objects over or accidental harm.',
        'misbehavior2': 'May exhibit frustration due to communication barriers or physical limitations.',
        'misbehavior3': 'Struggles with fine motor activities, resulting in refusal to engage in tasks requiring dexterity.',
        'misbehavior4': 'Can show aggression when overly frustrated or when unable to express needs effectively.',
        'misbehavior5': 'Frequent crying or vocal outbursts when in uncomfortable positions or unable to move as desired.'
    },
    'Sensory Processing Disorder': {
        'misbehavior1': 'Overreacts to touch, sounds, or lights, which can lead to withdrawal or aggression.',
        'misbehavior2': 'Avoids certain textures in food or clothing, which can be challenging during meals or dressing.',
        'misbehavior3': 'Has difficulty engaging in play that involves multiple sensory inputs.',
        'misbehavior4': 'May become overwhelmed easily in busy environments, resulting in shutdowns or meltdowns.',
        'misbehavior5': 'Exhibits extreme behaviors (either underreactive or overreactive) to normal stimuli.'
    }
}


disorder = st.sidebar.selectbox(
    '아이가 어떤 장애를 가지고 있나요?',
    ['ASD', 'ADHD', 'Down Syndrome', 'Cerebral Palsy', 'Sensory Processing Disorder']
)

if disorder:
    behavior = st.chat_input('영유아가 보이고 있는 행동을 입력해주세요!')
    
    if behavior:
        INPUT_FOR_SUMMARY = f""" \n
        Disorder child is going through : {disorder} \n
        
        Behavior one is showing : {behavior}
        """
        st.session_state.messages.append({'role': 'user', 'content': INPUT_FOR_SUMMARY})
        with st.chat_message('assistant'):
            st.markdown(INPUT_FOR_SUMMARY)
        
        with st.spinner('행동을 분석하고 교육 방향성에 대한 조언을 생성 중이에요!!'):
        
            BEHAVIOR_ANALYZER_PROMPT = """
            Summarize the misbehavior exhibited by a infant or toddler with specific disabilities in two sentences based on the input provided. The input will include details about the misbehavior being exhibited and the specific disabilities of the infant or toddler. Your summary should accurately capture the observed behavior while being concise and clear. Please ensure that the summary is sensitive and respectful to the individual's condition and circumstances.
            """

            summarized_action = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': BEHAVIOR_ANALYZER_PROMPT},
                    {'role': 'user', 'content': INPUT_FOR_SUMMARY}
                ]
            ).choices[0].message.content
            
            st.session_state.messages.append({'role': 'summary', 'content': summarized_action})
            
            action_embedding = get_embedding(summarized_action)
            max_similarity = -1
            most_similar_behavior = None
            
            for key1, value1 in disorder_behavior_DB.items():
                if key1 == disorder:
                    for key2, value2 in value1.items():
                        behavior_embedding = get_embedding(value2)
                        similarity = cosine_similarity(action_embedding, behavior_embedding)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_behavior = value2
                else:
                    pass
                
            st.session_state.messages.append({'role': 'DB', 'content': most_similar_behavior})
            
            BEHAVIOR_SUGGESTOR = """
            From now on, you're going to be an advisor to educators of disabled infants and toddlers. Your role is to provide guidance on how to handle the actions of disabled children and prevent any inappropriate behavior. You'll receive information about the specific behavior, the disorder the child has, and the typical behavioral characteristics associated with that disorder. Based on this information, you should offer advice on interventions that can be used to help educate the child with the disorder and address their misbehavior effectively. Remember, your focus is on providing guidance and support to educators in handling these situations.
            """
            
            ADVICE_INPUT = f"Disorder : {disorder} \n Todller's action : {summarized_action} \n Most similar behavior : {most_similar_behavior}"
            
            final_advice = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                messages=[
                    {'role': 'system', 'content': BEHAVIOR_SUGGESTOR},
                    {'role': 'user', 'content': ADVICE_INPUT}
                ]
            ).choices[0].message.content
            
            st.session_state.messages.append({'role': 'final_advice', 'content': final_advice})
            
            with st.expander('조언을 확인하세요!!'):
                st.markdown(f"📝 행동 요약 : {summarized_action}")
                st.markdown(f"🩺 유사 행동 증상 : {most_similar_behavior}")
                st.markdown(f"👨‍⚕️ 조언 : {final_advice}")