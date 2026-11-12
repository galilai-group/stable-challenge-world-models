"use strict";
// Browser version of test_submission.py. The submission page runs it in a Web Worker on the chosen file and only
// uploads files that pass, so a broken model doesn't use up one of the participant's daily attempts.
//
// Keep the checks in step with test_submission.py and evaluate.py (load_model, run_batches), which stay the reference.
// It adds a first check that the file is ONNX at all; the throughput estimate runs only locally.
//
// Protocol: the page posts {file, max_bytes, base_url}; this script posts
//   {type: "steps", steps: [title, ...], command}       once, first
//   {type: "step", index, status: "running"|"ok"|"fail", note?}
//   {type: "warn", index, message}
//   {type: "done", ok, message?}                        the file passed (ok) or must be fixed (message says how)
//   {type: "unavailable", message}                      the check itself could not run; the page doesn't block

const ORT_VERSION = "1.22.0";  // closest onnxruntime-web release to the evaluator's onnxruntime 1.22.1
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const COMMAND = "uv run test_submission.py model.onnx";
const BATCH = 64, HIST = 3, SIZE = 64, HORIZON = 5, ACTION_DIM = 2, SKIP = 5, MAX_DIM = 1024, OPSET_MIN = 17;
const FRAMES_SHAPE = [BATCH, HIST, 3, SIZE, SIZE], ACTIONS_SHAPE = [BATCH, HORIZON, ACTION_DIM * SKIP];
const FRAME_SIZE = HIST * 3 * SIZE * SIZE, ACTION_SIZE = HORIZON * ACTION_DIM * SKIP;
// The evaluator's onnxruntime 1.22.1 reads ONNX IR versions up to 10 and opsets up to 22. They are checked before
// loading because onnxruntime-web 1.22 reports load errors without a message.
const MAX_IR = 10, MAX_OPSET = 22;
const EXPORT_HINT = "export() in example_train.py exports a world model in the right format.";

class Failed extends Error {}  // a problem with the model; the message says how to fix it
class Unavailable extends Error {}  // the check itself could not run in this browser
class SubmissionError extends Error {}  // raised by run, like evaluate.SubmissionError

const post = message => self.postMessage(message);

function warn(ctx, message) {
  ctx.warnings.push(message);
  post({type: "warn", index: ctx.step, message});
}

const pyShape = dims => "(" + dims.join(", ") + (dims.length === 1 ? ",)" : ")");
const pyList = dims => "[" + dims.map(d => typeof d === "string" ? `'${d}'` : String(d)).join(", ") + "]";
const mib = bytes => (bytes / 1024 ** 2).toLocaleString("en-US", {minimumFractionDigits: 1, maximumFractionDigits: 1});
const num = x => x.toLocaleString("en-US", {maximumSignificantDigits: 6});

// --------------------------------------------------------------------------------------------- ONNX protobuf

// Reads just the parts of an ONNX ModelProto the checks need, skipping weight data, so a 20 MiB file parses quickly.
// Field numbers are from onnx.proto.
class Proto {
  constructor(bytes, start = 0, end = bytes.length) { this.b = bytes; this.pos = start; this.end = end; }

  varint() {
    let result = 0n, shift = 0n;
    for (;;) {
      if (this.pos >= this.end || shift > 63n) throw new Error("truncated or malformed varint");
      const byte = this.b[this.pos++];
      result |= BigInt(byte & 0x7f) << shift;
      if (!(byte & 0x80)) return BigInt.asIntN(64, result);
      shift += 7n;
    }
  }

  // Yields [field number, wire type, value], where value is a number for varints and a sub-reader for
  // length-delimited fields. Other wire types are skipped.
  *fields() {
    while (this.pos < this.end) {
      const key = Number(this.varint()), field = key >>> 3, wire = key & 7;
      if (wire === 0) yield [field, wire, Number(this.varint())];
      else if (wire === 2) {
        const length = Number(this.varint());
        if (length < 0 || this.pos + length > this.end) throw new Error("truncated length-delimited field");
        yield [field, wire, new Proto(this.b, this.pos, this.pos + length)];
        this.pos += length;
      } else if (wire === 1 || wire === 5) {
        this.pos += wire === 1 ? 8 : 4;
        if (this.pos > this.end) throw new Error("truncated fixed-width field");
      } else throw new Error(`unsupported wire type ${wire}`);
    }
  }

  string() { return new TextDecoder().decode(this.b.subarray(this.pos, this.end)); }
}

function parseModel(bytes) {
  const model = {ir_version: null, opsets: {}, graph: null};
  for (const [field, wire, value] of new Proto(bytes).fields()) {
    if (field === 1 && wire === 0) model.ir_version = value;
    else if (field === 8 && wire === 2) {
      let domain = "", version = null;
      for (const [f, w, v] of value.fields()) {
        if (f === 1 && w === 2) domain = v.string();
        else if (f === 2 && w === 0) version = v;
      }
      model.opsets[domain || "ai.onnx"] = version;
    } else if (field === 7 && wire === 2) model.graph = parseGraph(value);
  }
  return model;
}

function parseGraph(proto) {
  const graph = {nodes: 0, inputs: [], outputs: [], initializers: []};
  for (const [field, wire, value] of proto.fields()) {
    if (wire !== 2) continue;
    if (field === 1) graph.nodes++;
    else if (field === 5) {
      let name = "", external = false;
      for (const [f, w, v] of value.fields()) {
        if (f === 8 && w === 2) name = v.string();
        else if (f === 14 && w === 0) external = v === 1;  // data_location == EXTERNAL
      }
      graph.initializers.push({name, external});
    } else if (field === 11 || field === 12) (field === 11 ? graph.inputs : graph.outputs).push(parseValueInfo(value));
  }
  return graph;
}

function parseValueInfo(proto) {
  const info = {name: "", tensor: false, dtype: null, dims: null};
  for (const [field, wire, value] of proto.fields()) {
    if (field === 1 && wire === 2) info.name = value.string();
    else if (field === 2 && wire === 2) {
      for (const [f, w, tensor] of value.fields()) {
        if (f !== 1 || w !== 2) continue;  // TypeProto.tensor_type
        info.tensor = true;
        for (const [tf, tw, tv] of tensor.fields()) {
          if (tf === 1 && tw === 0) info.dtype = DTYPES[tv] || `TYPE_${tv}`;
          else if (tf === 2 && tw === 2) {
            info.dims = [];
            for (const [sf, sw, dim] of tv.fields()) {
              if (sf !== 1 || sw !== 2) continue;
              let d = null;
              for (const [df, dw, dv] of dim.fields()) {
                if (df === 1 && dw === 0) d = dv;
                else if (df === 2 && dw === 2 && d === null) d = dv.string() || null;
              }
              info.dims.push(d);
            }
          }
        }
        info.dtype ??= "UNDEFINED";
      }
    }
  }
  return info;
}

const DTYPES = ["UNDEFINED", "FLOAT", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING", "BOOL",
  "FLOAT16", "DOUBLE", "UINT32", "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ",
  "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "UINT4", "INT4", "FLOAT4E2M1", "FLOAT8E8M0", "UINT2", "INT2"];

// --------------------------------------------------------------------------------------------- 1. file

function zipNames(bytes) {
  // The end-of-central-directory record is in the last 64 KiB; it points at the list of file names.
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) {
    if (view.getUint32(i, true) !== 0x06054b50) continue;
    const names = [], count = view.getUint16(i + 10, true);
    let p = view.getUint32(i + 16, true);
    for (let k = 0; k < count; k++) {
      if (p + 46 > bytes.length || view.getUint32(p, true) !== 0x02014b50) return null;
      const n = view.getUint16(p + 28, true), extra = view.getUint16(p + 30, true), comment = view.getUint16(p + 32, true);
      names.push(new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + n)));
      p += 46 + n + extra + comment;
    }
    return names;
  }
  return null;
}

function sniff(bytes, head) {
  const starts = prefix => typeof prefix === "string"
    ? [...prefix].every((c, i) => head[i] === c.charCodeAt(0)) : prefix.every((b, i) => head[i] === b);
  const pytorch = "This looks like a PyTorch checkpoint (torch.save / TorchScript), not an ONNX model. Load your " +
    "model in PyTorch and export it to ONNX, as export() in example_train.py does.";
  if (starts("version https://git-lfs"))
    return "This is a Git LFS pointer file, not the model itself. Run `git lfs pull` (or download the file " +
      "directly) to get the real model.";
  const firstNonSpace = head.find(b => ![0x20, 0x09, 0x0a, 0x0b, 0x0c, 0x0d].includes(b));
  if (firstNonSpace === 0x3c)
    return "This is an HTML/XML page, not a model. This usually happens when downloading from a share link " +
      "(Google Drive, Colab, Dropbox, ...) saves the web page instead of the file. Download the file " +
      "itself, for example from the browser's download button.";
  if (starts("PK")) {
    const names = zipNames(bytes);
    if (names === null) return "This is a damaged zip archive, not an ONNX model.";
    const onnx = names.filter(n => n.toLowerCase().endsWith(".onnx"));
    if (onnx.length)
      return `This is a zip archive containing '${onnx[0]}'. Unzip it and check/submit the .onnx file itself.`;
    if (names.some(n => ["data.pkl", "constants.pkl", "version"].some(s => n.endsWith(s)))) return pytorch;
    return "This is a zip archive, not an ONNX model. Submit the .onnx file itself.";
  }
  if (starts([0x1f, 0x8b])) return "This is a gzip-compressed file. Decompress it (gunzip) and submit the .onnx file itself.";
  if (head[0] === 0x80 && [2, 3, 4, 5].includes(head[1]))
    return pytorch.replace("(torch.save / TorchScript)", "or another Python pickle");
  if (starts([0x89, 0x48, 0x44, 0x46]))
    return "This is an HDF5 file (for example a Keras .h5 model), not ONNX. Convert it with tf2onnx, or export " +
      "your PyTorch model as export() in example_train.py does.";
  if (starts([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]))  // \x93NUMPY
    return "This is a NumPy array file (.npy), not an ONNX model.";
  if (head.length >= 10 && head[8] === 0x7b && head[9] === 0x22)
    return "This looks like a safetensors file, which holds only weights. Load the weights into your model " +
      "in PyTorch and export it as export() in example_train.py does.";
  try {
    const text = new TextDecoder("utf-8", {fatal: true}).decode(head);
    if (/^(?:\P{C}|\s)*$/u.test(text))
      return `This is a text file, not an ONNX model. It starts with: '${text.slice(0, 80)}'`;
  } catch { /* not UTF-8 text */ }
  return null;
}

async function checkFile(ctx) {
  const {file} = ctx;
  if (file.size === 0)
    throw new Failed("The file is empty (0 bytes). The export or the copy did not finish; export the model again.");
  ctx.bytes = new Uint8Array(await file.arrayBuffer());
  const problem = sniff(ctx.bytes, ctx.bytes.subarray(0, 512));
  if (problem) throw new Failed(problem);
  if (file.size > ctx.maxBytes)
    throw new Failed(`The file is ${mib(file.size)} MiB, over the ${num(ctx.maxBytes / 1024 ** 2)} MiB upload ` +
      `limit, so the submission page will reject it. Use a smaller model (about ` +
      `${num(ctx.maxBytes / 1024 ** 2 / 4)} million float32 parameters at most).`);
  const dot = file.name.lastIndexOf(".");
  const suffix = dot > 0 ? file.name.slice(dot) : "";
  if (suffix.toLowerCase() !== ".onnx")
    warn(ctx, `The file name ends in '${suffix || "(no extension)"}' rather than '.onnx'. That's fine for ` +
      `evaluation, but double-check that this is the file you meant.`);
  return `${file.name}, ${mib(file.size)} MiB`;
}

// --------------------------------------------------------------------------------------------- 2. graph

function checkGraph(ctx) {
  let model;
  try {
    model = parseModel(ctx.bytes);
  } catch (e) {
    throw new Failed(`The file is not a readable ONNX model (${e.message}). It may be truncated by an interrupted ` +
      `export, copy or download, or not be ONNX at all. Export it again.`);
  }
  const graph = model.graph || {nodes: 0, inputs: [], outputs: [], initializers: []};
  if (!graph.nodes && !graph.outputs.length)
    throw new Failed("The file parses but contains no model graph, so it is probably not an ONNX model. Export it " +
      "again. " + EXPORT_HINT);
  const external = graph.initializers.filter(t => t.external).map(t => t.name);
  if (external.length)
    throw new Failed(`The weights (${external.length} tensors, e.g. '${external[0]}') are stored in separate files ` +
      `next to the .onnx (ONNX 'external data'), but only the .onnx file is uploaded. Merge them into one file ` +
      `with:\n    import onnx; onnx.save(onnx.load('${ctx.file.name}'), 'merged.onnx')\n` +
      `This only works for models under 2 GB.`);
  // test_submission.py checks the opset after running the model; it needs no runtime, so check it first here.
  const opset = model.opsets["ai.onnx"] ?? null;
  ctx.opset = opset;
  if (opset !== null && opset < OPSET_MIN)
    throw new Failed(`The model uses ONNX opset ${opset}, but the minimum is ${OPSET_MIN}. Export with ` +
      `opset_version=${OPSET_MIN} or newer. ` + EXPORT_HINT);
  if (opset !== null && opset > MAX_OPSET)
    throw new Failed(`The model uses ONNX opset ${opset}, newer than the evaluator's onnxruntime supports (at most ` +
      `${MAX_OPSET}). Export with opset_version=${OPSET_MIN}.`);
  if (model.ir_version !== null && model.ir_version > MAX_IR)
    throw new Failed(`The model file format (ONNX IR version ${model.ir_version}) is newer than the evaluator's ` +
      `onnxruntime reads (at most ${MAX_IR}), usually because a newer onnx/torch wrote it. Lower it with:\n` +
      `    import onnx; m = onnx.load('${ctx.file.name}'); m.ir_version = ${MAX_IR}; onnx.save(m, '${ctx.file.name}')`);
  return `${graph.nodes} nodes, opset ${opset ?? "unknown"} (minimum ${OPSET_MIN})`;
}

// --------------------------------------------------------------------------------------------- 3. load

// onnxruntime-web wraps errors in its C API call and the C++ source location; keep only the message itself.
const ortMessage = e => String(e?.message || e).trim()
  .replace(/^(?:Can't create a session\.|failed to call OrtRun\(\)\.)\s*ERROR_CODE: \d+, ERROR_MESSAGE: /, "")
  .replace(/^\S+\.(?:h|cc|cpp):\d+ [^(]*\([^)]*\)(?: const)? /, "");

// A one-node Identity model, loaded first so a runtime that can't start is never blamed on the participant's model.
const TINY_MODEL = "CAgSADo3ChAKAXgSAXkiCElkZW50aXR5EgF0Wg8KAXgSCgoICAESBAoCCAFiDwoBeRIKCggIARIECgIIAUIECgAQEQ==";

let runtime = null;
function startRuntime() {
  runtime ??= (async () => {
    importScripts(ORT_URL + "ort.wasm.min.js");
    ort.env.wasm.wasmPaths = ORT_URL;
    ort.env.wasm.numThreads = 1;  // threads need cross-origin isolation, which the submission page doesn't have
    ort.env.logLevel = "error";
    const tiny = Uint8Array.from(atob(TINY_MODEL), c => c.charCodeAt(0));
    await (await ort.InferenceSession.create(tiny)).release();
  })();
  return runtime;
}

// Same rules as evaluate.load_model.
async function checkLoad(ctx) {
  try {
    await startRuntime();
  } catch (e) {
    throw new Unavailable(`onnxruntime-web could not start in this browser (${e?.message || e}).`);
  }
  let sess;
  try {
    sess = await ort.InferenceSession.create(ctx.bytes, {executionProviders: ["wasm"]});
  } catch (e) {
    const msg = ortMessage(e), lower = msg.toLowerCase();
    let hint;
    if (/^\d*$/.test(msg))  // onnxruntime-web 1.22 sometimes throws only an internal error code
      throw new Failed(`onnxruntime could not load the model, and gave no details in the browser. Run ${COMMAND} on ` +
        `your computer to see the full error.`);
    if (msg.includes("IR version"))
      hint = `The model file format (ONNX IR version) is newer than the evaluator's onnxruntime reads, usually ` +
        `because a newer onnx/torch wrote it. Lower it with:\n    import onnx; m = onnx.load('${ctx.file.name}'); ` +
        `m.ir_version = 10; onnx.save(m, '${ctx.file.name}')`;
    else if (lower.includes("opset") && (lower.includes("support") || lower.includes("version")))
      hint = `The model uses ONNX opset ${ctx.opset ?? "unknown"}, newer than the evaluator's onnxruntime ` +
        `supports. Export with opset_version=${OPSET_MIN}.`;
    else if (msg.includes("No Op registered") || lower.includes("not a registered") || lower.includes("custom") ||
             lower.includes("domain"))
      hint = "The model uses an operator that onnxruntime doesn't have (a custom op or a non-standard domain). " +
        "Evaluation uses plain onnxruntime, so the model must use standard ONNX ops; replace the custom layer " +
        "(e.g. a fused CUDA kernel) with plain PyTorch ops before export.";
    else hint = "Re-export the model, or simplify the layer named in the error above.";
    throw new Failed(`Could not load the ONNX model:\n    ${msg}\n${hint}`);
  }
  const ins = sess.inputMetadata, outs = sess.outputMetadata;
  const names = list => list.map(v => `'${v.name}'`).join(", ") || "none";
  if (ins.length !== 2 || outs.length !== 1)
    throw new Failed(`ONNX model must have exactly two inputs (frames, actions) and one output (z), but this one has ` +
      `${ins.length} input(s) (${names(ins)}) and ${outs.length} output(s) (${names(outs)}). ` + EXPORT_HINT);
  const sorted = ins.map(v => v.name).sort();
  if (sorted[0] !== "actions" || sorted[1] !== "frames")
    throw new Failed(`ONNX inputs must be named 'frames' and 'actions'; got ${pyList(sorted)}. Export with ` +
      `input_names=["frames", "actions"]. ` + EXPORT_HINT);
  for (const input of ins) {
    const want = input.name === "frames" ? FRAMES_SHAPE : ACTIONS_SHAPE;
    if (!input.isTensor || input.type !== "float32") {
      const got = input.isTensor ? input.type : "a non-tensor value";
      throw new Failed(`Input '${input.name}' must be float32, not ${got}. Export the model in float32 (no ` +
        `model.half()) with float32 example inputs.`);
    }
    const shape = input.shape;
    if (shape.length !== want.length || shape.some((g, i) => typeof g === "number" && g !== want[i]))
      throw new Failed(`Input '${input.name}' must accept shape ${pyShape(want)}; it declares ${pyList(shape)}. ` +
        `Export with example inputs torch.rand${pyShape(FRAMES_SHAPE)} for frames and ` +
        `torch.randn${pyShape(ACTIONS_SHAPE)} for actions. ` + EXPORT_HINT);
  }
  ctx.session = sess;
  return `inputs 'frames' ${pyShape(FRAMES_SHAPE)} and 'actions' ${pyShape(ACTIONS_SHAPE)}; onnxruntime-web ` +
    `${ORT_VERSION} in this browser`;
}

// --------------------------------------------------------------------------------------------- 4-8. running

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

function floats(tensor) {
  if (tensor.type === "float16" && tensor.data instanceof Uint16Array) return Float64Array.from(tensor.data, halfToFloat);
  return Float64Array.from(tensor.data, Number);
}

function explainSubmissionError(e) {
  const msg = e.message;
  if (msg.startsWith("Output shape"))
    return `${msg}. The model must return the embedding of the observation followed by the ${HORIZON} predicted ` +
      `embeddings, [${BATCH}, ${HORIZON + 1}, D], as WorldModel.forward in example_train.py does.`;
  if (msg.startsWith("Embedding width"))
    return `${msg}. Make the last layer narrower, or project the embedding to at most ${MAX_DIM} features.`;
  if (msg.includes("NaN or infinite"))
    return `${msg}. Look for divisions by zero, log(0), normalising by a zero standard deviation (add a small ` +
      `epsilon), or float16 overflow.`;
  if (msg.startsWith("Model failed"))
    return `${msg}\nThe model crashed while running. Check that it accepts the input shapes above; export with ` +
      `example inputs of exactly those shapes.`;
  return msg;
}

// Same as evaluate.run_batches for one full batch: frames float32 [BATCH * FRAME_SIZE] in [0, 1] and actions
// float32 [BATCH * ACTION_SIZE] -> z as BATCH rows of [(HORIZON + 1) * D] values, and D.
async function run(sess, frames, actions) {
  let z;
  try {
    const result = await sess.run({frames: new ort.Tensor("float32", frames, FRAMES_SHAPE),
                                   actions: new ort.Tensor("float32", actions, ACTIONS_SHAPE)});
    z = result[sess.outputNames[0]];
  } catch (e) {
    throw new SubmissionError(`Model failed on frames ${pyShape(FRAMES_SHAPE)} and actions ` +
      `${pyShape(ACTIONS_SHAPE)}: ${ortMessage(e)}`);
  }
  if (z.dims.length !== 3 || z.dims[0] !== BATCH || z.dims[1] !== HORIZON + 1)
    throw new SubmissionError(`Output shape ${pyShape(z.dims)}, expected (${BATCH}, ${HORIZON + 1}, D)`);
  const dim = z.dims[2];
  if (!(dim >= 1 && dim <= MAX_DIM)) throw new SubmissionError(`Embedding width ${dim}, expected between 1 and ${MAX_DIM}`);
  const values = floats(z);
  if (!values.every(Number.isFinite)) throw new SubmissionError("Embeddings contain NaN or infinite values");
  return {values, dim};
}

async function runOrFail(ctx, frames, actions) {
  try {
    return await run(ctx.session, frames, actions);
  } catch (e) {
    if (e instanceof SubmissionError) throw new Failed(explainSubmissionError(e));
    throw e;
  }
}

// Deterministic uniform numbers from a seed (xorshift32).
function uniforms(n, seed, lo, hi) {
  return Float32Array.from({length: n}, () => {
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    return lo + (hi - lo) * ((seed >>> 0) + 0.5) / 2 ** 32;
  });
}

// Like test_submission.py: random frames in [0, 1] (whole pixels, as the evaluator feeds them), actions in [-1, 1].
const randomFrames = seed => uniforms(BATCH * FRAME_SIZE, seed, 0, 1).map(v => Math.floor(v * 256) / 255);
const randomActions = seed => uniforms(BATCH * ACTION_SIZE, seed, -1, 1);

const near = (a, b) => Math.abs(a - b) <= 1e-4 + 1e-3 * Math.abs(b);

// Are z[:, steps] the same in both outputs (all samples)?
function sameSteps(z1, z2, steps) {
  const {dim} = z1, row = (HORIZON + 1) * dim;
  for (let i = 0; i < BATCH; i++)
    for (const k of steps)
      for (let j = 0; j < dim; j++)
        if (!near(z2.values[i * row + k * dim + j], z1.values[i * row + k * dim + j])) return false;
  return true;
}

const range = (from, to) => Array.from({length: to - from}, (_, i) => from + i);

async function checkOutput(ctx) {
  ctx.f = randomFrames(1);
  ctx.a = randomActions(2);
  ctx.z = await runOrFail(ctx, ctx.f, ctx.a);
  return `output (${BATCH}, ${HORIZON + 1}, ${ctx.z.dim}), finite`;
}

async function checkDeterministic(ctx) {
  const again = await runOrFail(ctx, ctx.f, ctx.a);
  if (!ctx.z.values.every((v, i) => v === again.values[i]))
    throw new Failed("The same input gave different outputs. The model probably contains randomness such as " +
      "dropout; call model.eval() before exporting.");
  return "the same input gives the same output";
}

async function checkIndependent(ctx) {
  // reverse the order of the samples in the batch
  const flip = (x, size) => {
    const out = new Float32Array(x.length);
    for (let i = 0; i < BATCH; i++) out.set(x.subarray(i * size, (i + 1) * size), (BATCH - 1 - i) * size);
    return out;
  };
  const z = await runOrFail(ctx, flip(ctx.f, FRAME_SIZE), flip(ctx.a, ACTION_SIZE));
  const row = (HORIZON + 1) * ctx.z.dim;
  for (let i = 0; i < BATCH; i++)
    for (let j = 0; j < row; j++)
      if (!near(z.values[(BATCH - 1 - i) * row + j], ctx.z.values[i * row + j]))
        throw new Failed("A sample's output depends on the other samples in its batch. This usually means batch " +
          "norm was exported in training mode; call model.eval() before exporting.");
  return "each sample processed independently of its batch";
}

async function checkObservation(ctx) {
  ctx.a2 = randomActions(3);
  ctx.z2 = await runOrFail(ctx, ctx.f, ctx.a2);
  if (!sameSteps(ctx.z, ctx.z2, [0]))
    throw new Failed("z[:, 0] changes with the actions; it must embed the observation alone. The evaluator embeds " +
      "real observations (and goals) with z[:, 0], so compute it from the frames only, before the predictor uses " +
      "any action.");
  return "z[:, 0] does not depend on the actions";
}

async function checkCausal(ctx) {
  for (let k = 1; k < HORIZON; k++) {
    const a3 = Float32Array.from(ctx.a);
    for (let i = 0; i < BATCH; i++)
      for (let j = k * ACTION_DIM * SKIP; j < ACTION_SIZE; j++) a3[i * ACTION_SIZE + j] = ctx.a2[i * ACTION_SIZE + j];
    const z = await runOrFail(ctx, ctx.f, a3);
    if (!sameSteps(ctx.z, z, range(0, k + 1)))
      throw new Failed(`Not causal: z[:, ${k}] changes when actions[:, ${k}:] change. The prediction k steps ahead ` +
        `may use only the first k action steps, actions[:, :k]: roll the predictor out one step at a time, as ` +
        `WorldModel.rollout in example_train.py does.`);
  }
  if (sameSteps(ctx.z, ctx.z2, range(1, HORIZON + 1)))
    warn(ctx, "The predictions z[:, 1:] ignore the actions: the model has no action-conditioned dynamics. It is " +
      "accepted, but expect a low score.");
  return "causal: z[:, k] uses only actions[:, :k]";
}

// --------------------------------------------------------------------------------------------- main

const STEPS = [
  ["File is an ONNX model", checkFile],
  ["ONNX graph is valid", checkGraph],
  ["Loads, with float32 inputs 'frames' (64, 3, 3, 64, 64) and 'actions' (64, 5, 10)", checkLoad],
  ["Returns finite (64, 6, D) embeddings, 1 ≤ D ≤ 1024", checkOutput],
  ["Deterministic", checkDeterministic],
  ["Each sample processed independently of its batch", checkIndependent],
  ["z[:, 0] does not depend on the actions", checkObservation],
  ["Causal: z[:, k] uses only actions[:, :k]", checkCausal],
];

async function main({file, max_bytes, base_url}) {
  post({type: "steps", steps: STEPS.map(([title]) => title), command: COMMAND});
  startRuntime().catch(() => {});  // download onnxruntime while the first checks run; failures surface in step 3
  const ctx = {file, maxBytes: max_bytes, baseUrl: base_url, warnings: [], step: 0};
  for (const [index, [, check]] of STEPS.entries()) {
    ctx.step = index;
    post({type: "step", index, status: "running"});
    try {
      const note = await check(ctx);
      post({type: "step", index, status: "ok", note});
    } catch (e) {
      if (e instanceof Unavailable) { post({type: "unavailable", message: e.message}); return; }
      post({type: "step", index, status: "fail"});
      post({type: "done", ok: false, message: e instanceof Failed ? e.message
        : `Unexpected error: ${e?.message || e}\nThis most likely comes from the model. If you think it's a bug ` +
          `in the check, run ${COMMAND} and report its output to the organizers.`});
      return;
    }
  }
  post({type: "done", ok: true});
}

self.onmessage = event => { main(event.data); };
