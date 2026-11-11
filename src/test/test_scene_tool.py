"""Check the tool protocol without launching a simulator or paying API calls."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assets'))
import scene_tool


def test_available_ids_and_seeded_plan():
    ids=scene_tool.execute({'action':'list_assets'})
    assert len(ids['satellites'])==26 and len(ids['arms'])==2
    assert 'satellite_6' not in ids['satellites']
    request={'action':'plan_scene','seed':42}
    one=scene_tool.execute(request);two=scene_tool.execute(request)
    assert one==two and one['plan']['task_ready'] is False


@pytest.mark.parametrize('payload',[
    {'action':'train'},
    {'action':'render_scene','run_id':'../../outside'},
    {'action':'render_scene','seed':True,'run_id':'a'},
    {'action':'plan_scene','shell':'rm -rf /'},
])
def test_invalid_requests_fail_before_simulator(payload):
    with pytest.raises(ValueError):scene_tool.execute(payload)


def test_duplicate_run_is_not_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(scene_tool,'ROOT',tmp_path)
    folder=tmp_path/'artifacts/agent_demo/existing';folder.mkdir(parents=True)
    marker=folder/'result.json';marker.write_text('preserve')
    with pytest.raises(FileExistsError):scene_tool.execute({'action':'render_scene','run_id':'existing'})
    assert marker.read_text()=='preserve'


def test_api_adapter_dispatch_and_tool_feedback_without_live_api(tmp_path,monkeypatch):
    import io
    import run_tool_agent
    prompt=tmp_path/'prompt.txt';prompt.write_text('Plan a random inspection scene')
    output=tmp_path/'trace.json'
    monkeypatch.setenv('API_KEY','test-only-not-a-real-key')
    monkeypatch.setattr(sys,'argv',['agent','--prompt',str(prompt),'--output',str(output)])
    sent=[]
    def fake_api(req,timeout):
        payload=json.loads(req.data);sent.append(payload)
        if len(sent)==1:
            message={'role':'assistant','content':None,'tool_calls':[{'id':'test_call_1','type':'function',
                'function':{'name':'spaceman_scene','arguments':json.dumps({'action':'plan_scene','seed':42})}}]}
        else:
            feedback=payload['messages'][-1]
            assert feedback['role']=='tool' and feedback['tool_call_id']=='test_call_1'
            assert json.loads(feedback['content'])['plan']['task_ready'] is False
            message={'role':'assistant','content':'Actual plan returned; no rendering requested'}
        return io.BytesIO(json.dumps({'choices':[{'message':message}]}).encode())
    monkeypatch.setattr(run_tool_agent.urllib.request,'urlopen',fake_api)
    assert run_tool_agent.main()==0
    text=output.read_text();assert 'test-only-not-a-real-key' not in text
    trace=json.loads(text);assert trace[1]['result']['ok']
