import streamlit as st
import pandas as pd
import plotly.express as px
from core import generate,validate,train,aggregate,scenario
st.set_page_config(page_title='TalentFlow AI',page_icon='◈',layout='wide')
st.title('TalentFlow AI | Workforce Intelligence')
st.caption('ZubairLabs • Organizational analytics • Research demonstration')
st.warning('Synthetic demo by default. Not for individual employment decisions. Scenario estimates are associations, not causal effects.')
@st.cache_data
def demo(): return generate()
@st.cache_resource
def demo_model(): return train(generate())
u=st.sidebar.file_uploader('Optional anonymized CSV',type='csv')
try:
    if u:
        if u.size>5_000_000: raise ValueError('Maximum upload is 5 MB')
        d=validate(pd.read_csv(u)); model,metrics,cm=train(d)
    else: d=demo(); model,metrics,cm=demo_model()
except Exception as e: st.error(str(e)); st.stop()
a=aggregate(d)
cols=st.columns(4)
for c,(k,v) in zip(cols,[('Records',len(d)),('Observed turnover',f'{d.left.mean():.1%}'),('Mean engagement',f'{d.engagement.mean():.1f}/5'),('Departments',d.department.nunique())]): c.metric(k,v)
tabs=st.tabs(['Overview','Scenario lab','Model evaluation','Export'])
with tabs[0]:
    st.subheader('Department-level insights (minimum 10 records)'); st.plotly_chart(px.bar(a,x='department',y='turnover',color='department'),use_container_width=True); st.plotly_chart(px.scatter(a,x='weekly_hours',y='engagement',size='records',color='department'),use_container_width=True); st.dataframe(a)
with tabs[1]:
    h=st.slider('Weekly hours adjustment',-10,10,0); e=st.slider('Engagement adjustment',-1.,1.,0.,.1); t=st.slider('Training hours adjustment',0,40,0); b,s=scenario(model,d,h,e,t); st.metric('Scenario mean model score',f'{s:.1%}',f'{(s-b)*100:+.2f} percentage points'); st.caption('Hypothetical model sensitivity only; not an estimate of intervention effects.')
with tabs[2]:
    for k,v in metrics.items(): st.metric(k,f'{v:.3f}' if isinstance(v,float) else v)
    st.plotly_chart(px.imshow(cm,x=['Predicted stay','Predicted leave'],y=['Actual stay','Actual leave'],text_auto=True),use_container_width=True); st.caption('Random stratified holdout; not a prospective validation.')
with tabs[3]:
    st.download_button('Download department summary',a.to_csv(index=False),'department_summary.csv','text/csv'); st.download_button('Download synthetic example',demo().to_csv(index=False),'synthetic_example.csv','text/csv')
