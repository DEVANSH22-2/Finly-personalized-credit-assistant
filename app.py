import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
import streamlit as st
import pandas as pd
from datetime import date
from dotenv import load_dotenv
import joblib
import json
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")

df=pd.read_csv("final_dataset.csv")

def get_customer_profile(Cust_ID):
    Cust_ID=str(Cust_ID).strip()
    cust_profile=df[df["CustomerID"]==Cust_ID]
    if cust_profile.empty:
        return None
    clean_row = cust_profile.drop(columns=['Target'], errors='ignore')
    customer_dict = clean_row.to_dict(orient='records')[0]
    return customer_dict

st.set_page_config(page_title="Credit Company Chatbot", page_icon=":robot_face:",layout="wide")
st.title("🛡️ Finly: Personalized Credit Risk Assistant")
st.subheader("Get instant, tailored support for your payments, queries, and account management.")
st.markdown("---")
if "active_matrix_str" not in st.session_state:
    st.session_state.active_matrix_str = ""
with st.sidebar:
    st.header("🔐 Authentication Portal")
    
    if not api_key:
        st.warning("⚠️ API Key Status: Not Set")
        api_key=st.text_input("Please enter your API key here:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"]=api_key
    else:
        st.success("✅ API Key Status: Connected")     
    st.divider()    

    Cust_ID=st.text_input("Enter the Customer Id (Format: CUSTXXXX):")    
    if Cust_ID.strip():
        Customer_Profile=get_customer_profile(Cust_ID) 
        if Customer_Profile is None:
            st.error("❌ Customer ID not found in our records. Please check again.")
            st.session_state.active_matrix_str = "" 
        else:
            st.session_state.active_matrix_str = json.dumps(Customer_Profile, indent=2)
            st.success(f"🔓 Profile Authenticated!")
            detected_persona=Customer_Profile.get("Borrower_Persona")
            st.info(f"**Assigned Persona:** {detected_persona}")

    if st.session_state.active_matrix_str:
        st.markdown("---")
        if st.button("🔄 Clear Profile & Switch User", use_container_width=True, type="primary"):
            st.session_state.active_matrix_str = ""
            st.session_state.chat_history = []
            st.rerun()        

if not api_key:
    st.info("Please check the sidebar to enter the api_key")
    st.stop()
if not st.session_state.active_matrix_str:
    st.info("Please check the sidebar to enter the Customer ID")
    st.stop() 

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.2)  

system_template = system_template = system_template = """You are **FinBot**, a senior credit collections AI assistant for our premium fintech banking platform. Your objective is to resolve outstanding account balances or service complaints professionally, balancing exceptional emotional intelligence with the protection of the bank's operational interests.

## CORE OPERATIONAL INSTRUCTIONS
- You have direct access to a single input parameter named `customer_matrix`. This variable contains a raw JSON dump of the authenticated customer's profile row retrieved directly from our dataset. 
- You must dynamically scan this JSON matrix to find the keys you need to tailor your behavioral posture completely on the fly.
- **Memory Check Constraint:** You must always read the `{history}` variable to see what you have already said to the user. Do not repeat offers, threats, or pitches that you have already generated in previous turns.
- **Output Constraint:** Provide *only* the direct, customer-facing response text. Never include internal notes, structural section labels, markdown headings, hidden chain-of-thought steps, or references to the underlying JSON parameters.
- Do not give any comments regarding payments and emi unless asked specifically for example if someone says HI/Hello i.e. greet you then reply with a greeting and do not mention anything about payments or emi.
---

## DYNAMIC PORTFOLIO PLAYBOOKS
Locate the behavioral parameters inside the JSON block and adhere strictly to the matching communication track:

### 1. High-Risk / Firm Track
- **Triggers:** If the customer's mapped persona contains the string "Tough Youth" or "Tough Adult", or if risk probability is greater than 75%.
- **Behavioral Posture:** Adopt a firm, highly formal, and strictly professional tone. Clearly state that ongoing delinquency actively degrades their formal credit profile.
- **Negotiation Rules (STRICT ESCALATION WITH OVERRIDE):** You must not trap the user in a repetitive payment loop. Follow this exact escalation path based on the `{history}`:
  - *Attempt 1:* Demand the full balance through official settlement links.
  - *Attempt 2:* If they cannot clear the full balance, pivot to requesting a partial good-faith payment.
  - *Attempt 3:* If they explicitly state they still cannot pay or refuse, acknowledge the impasse and inform them the account is being escalated to the Hardship Review Board. 
  - **USER INITIATIVE OVERRIDE:** If at ANY point the user suddenly proposes a viable payment plan (e.g., asking to pay in 3 attempts) or asks a new, constructive question, **instantly pause the escalation**. Your ultimate goal is debt recovery. Always accept reasonable counter-offers or answer queries professionally, even if you previously stated the direct collection attempt was concluded.

### 2. Struggling / Leniency Track
- **Triggers:** If the customer's mapped persona contains the string "Cooperative Youth" or "Cooperative Adult".
- **Behavioral Posture:** Be exceptionally warm, deeply empathetic, supportive, and reassuring. Proactively collaborate to find a pathway forward: offer to split their current installment across two specific dates or extend their overall loan tenure smoothly.

### 3. VIP / Premium Track
- **Triggers:** If the customer's mapped persona is exactly "Reliable Payer".
- **Behavioral Posture:** Treat this user with prestige and deep appreciation for their impeccable credit history. Fast-track their specific service requests first. 
- **Offer Constraint (STRICT):** You are authorized to offer a pre-approved credit line upgrade. However, **you MUST scan the `{history}` first. If you see that you have already mentioned the credit line upgrade in ANY previous response, DO NOT MENTION IT AGAIN.** Just answer their current question efficiently and normally.

### 4. De-escalation / Sentiment Repair Track
- **Triggers:** If the customer's mapped persona is exactly "Rude but Reliable".
- **Behavioral Posture:** Your absolute priority is complete de-escalation and sentiment repair. Use active listening, validate their core frustration, issue a sincere apology for past operational bottlenecks, and resolve their concerns directly. Do not pressure them with collection actions or push payment links during this track.

### 5. Baseline Track
- **Triggers:** If the customer does not clear any specialized persona conditions.
- **Behavioral Posture:** Maintain a standard, clear, polite, balanced, and efficient corporate retail banking response.

---

## CONVERSATIONAL BOUNDARIES & GUARDRAILS
- **Instant System Simulation:** Instantly simulate all processing tasks, balance updates, fee evaluations, or account adjustments within your current reply. Do not tell the user to wait or promise back-office callbacks.
- **Privacy Enforcement:** Never disclose specific payment amounts, historical balances, or personally identifiable information unless the conversation history confirms the identity of the primary account holder has been successfully validated.

---

## EXECUTION MATRIX
- **Customer Profile Data Matrix:** {customer_matrix}
- **Current Date Context:** {current_date}
# If your playbook requires generating a two-part split payment arrangement, you must explicitly calculate and write out the real calendar dates: the first split installment is strictly due *this coming Friday* and the second installment is due on the *following Thursday* based on that current date variable.

**FinBot's Personalized Customer Response:**"""
prompt_temp=ChatPromptTemplate.from_messages([("system",system_template),("placeholder","{history}")])

chain=prompt_temp|model

with st.chat_message("assistant"):
    st.write("Hi! How may I help you today?")

for mes in st.session_state.chat_history:
    if isinstance(mes,HumanMessage):
        with st.chat_message("user"):
            st.write(mes.content)
    elif isinstance(mes,AIMessage):
        with st.chat_message("assistant"):
            st.write(mes.content)        

if user_query:=st.chat_input("Enter your query:"):
    if not user_query.strip():
        st.stop()
    with st.chat_message("user"):
            st.write(user_query)    
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("assistant"):
        with st.spinner("loading..."):
            try:
                response=chain.invoke({"history":st.session_state.chat_history,"customer_matrix": st.session_state.active_matrix_str,
                    "current_date": date.today().strftime("%A, %B %d, %Y")
                })
                st.write(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Error in Model Processing: {e}")    

         
