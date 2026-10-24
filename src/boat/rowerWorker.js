// Builds the rower's meshes off the main thread (see rower.js) and hands back typed arrays.
import { buildRowerArrays } from './rower.js';

self.onmessage = () => {
  const a = buildRowerArrays();
  const transfer = [];
  for (const k of ['body', 'head', 'handR', 'handL']) { a[k].data.forEach((d) => transfer.push(d.buffer)); transfer.push(a[k].idx.buffer); }
  self.postMessage(a, transfer);
};
