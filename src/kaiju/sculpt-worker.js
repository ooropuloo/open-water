// kaiju: builds the sculpt off the main thread (about a second of CPU work).
import { sculpt } from './sculpt.js';

self.onmessage = (e) => {
  const m = sculpt(e.data || {});
  m.low = sculpt({ cell: 1.0, low: true });
  self.postMessage(m, [m.pos.buffer, m.nrm.buffer, m.aux.buffer, m.idx.buffer, m.fins.buffer, m.low.pos.buffer, m.low.nrm.buffer, m.low.aux.buffer, m.low.idx.buffer]);
};
