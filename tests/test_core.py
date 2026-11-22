import pytest
from core import generate,validate,train,aggregate,scenario

def test_reproducible(): assert generate().equals(generate())
def test_schema(): assert len(validate(generate()))==1800
def test_missing():
    with pytest.raises(ValueError): validate(generate().drop(columns='left'))
def test_model():
    m,v,c=train(generate()); assert 0<=v['ROC AUC']<=1 and c.sum()==450
def test_aggregate(): assert aggregate(generate()).records.sum()==1800
def test_scenario():
    d=generate(); m,_,_=train(d); b,s=scenario(m,d); assert b==s
