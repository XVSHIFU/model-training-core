import numpy as np

from upload_judge.decision import Thresholds
from upload_judge.judge import UploadJudge


class DummyClassifier:
    thresholds = Thresholds()
    model_version = "dummy"

    def predict_success_proba(self, events):
        return np.asarray([0.9 if "model_success" in event.response.body else 0.05 for event in events])


def test_single_batch_parity_on_100_synthetic_events(make_event):
    events = []
    for index in range(100):
        if index % 4 == 0:
            body = '{"success":false}'
        elif index % 4 == 1:
            body = '{"success":true,"url":"/uploads/a.jpg"}'
        elif index % 4 == 2:
            body = '{"success":true,"note":"model_success"}'
        else:
            body = '{"message":"failed probability"}'
        events.append(make_event(body, event_id=f"evt-{index}", filename="a.jpg"))
    judge = UploadJudge(classifier=DummyClassifier())
    singles = [judge.judge(event).to_dict() for event in events]
    batch = [result.to_dict() for result in judge.judge_batch(events)]
    assert singles == batch


def test_single_verdict_output_contract(make_event):
    judge = UploadJudge(classifier=DummyClassifier())
    result = judge.judge(make_event('{"success":true,"fileId":"42"}', filename="part.step")).to_dict()
    assert result["verdict"] == "confirmed_upload_success"
    assert "upload_outcome" not in result
    assert "threat_level" not in result
