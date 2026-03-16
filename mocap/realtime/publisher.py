import json

import numpy as np
import zmq

ZMQ_HEADER_SIZE = 1280


class ZMQPublisher:
    def __init__(self, addr, topic="pose", header_size=ZMQ_HEADER_SIZE):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(addr)

        self.topic = topic
        self.topic_bytes = topic.encode("utf-8")
        self.header_size = header_size
        self.frame_idx = 0
        self.header_bytes = self._build_header(
            [
                {"name": "frame_index", "dtype": "i64", "shape": [1]},
                {"name": "body_quat", "dtype": "f64", "shape": [1, 4]},  # WXYZ format
                {"name": "smpl_joints", "dtype": "f64", "shape": [1, 24, 3]},
                {"name": "smpl_pose", "dtype": "f64", "shape": [1, 21, 3]},
            ]
        )

    def _build_header(self, fields, version=2, count=1):
        header = {"v": version, "endian": "le", "count": count, "fields": fields}
        return json.dumps(header, separators=(",", ":")).encode("utf-8").ljust(self.header_size, b"\0")

    def publish(self, body_quat, smpl_joints, smpl_pose):
        data = (
            self.topic_bytes
            + self.header_bytes
            + np.array([self.frame_idx], dtype=np.int64).tobytes()
            + np.asarray(body_quat, dtype=np.float64).reshape(1, 4).tobytes()
            + np.asarray(smpl_joints, dtype=np.float64).reshape(1, 24, 3).tobytes()
            + np.asarray(smpl_pose, dtype=np.float64).reshape(1, 21, 3).tobytes()
        )

        self.sock.send(data)
        self.frame_idx += 1

    def close(self):
        self.sock.close(0)
        self.ctx.term()
