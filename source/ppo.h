#pragma once
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

// ─────────── Dimensions ───────────
static const int STATE_DIM         = 181;
static const int ACTION_DIM        = 17;
static const int VAULT_ACTION_DIM  = 16;
static const int VAULT_STATE_DIM   = 174;  // state[0..173] → vault trunk
static const int SETUP_STATE_DIM   = 7;    // state[174..180] → setup head

static const int H1  = 256, H2  = 256;    // vault trunk hidden dims
static const int SH1 =  64, SH2 =  64;    // setup head hidden dims

// ─────────── Hyperparameters ───────────
static const int   ROLLOUT_LEN             = 512;
static const int   MIN_EPISODES_PER_UPDATE = 4;
static const int   BATCH_SIZE              = 32;
static const int   PPO_EPOCHS              = 4;
static const float PPO_LR                  = 3e-4f;
static const float PPO_GAMMA               = 0.99f;
static const float PPO_LAMBDA              = 0.95f;
static const float PPO_CLIP_EPS            = 0.2f;
static const float PPO_ENTROPY_COEF        = 0.01f;
static const float PPO_VALUE_COEF          = 0.5f;
static const int   MAX_EPISODE_STEPS       = 500;

static const float LOG_2PI = 1.8378770664f; // log(2*pi)

// ─────────── 3-layer tanh MLP, linear output ───────────
struct MLP
{
    int d0, d1, d2, d3;

    std::vector<float> W0, b0, W1, b1, W2, b2;
    std::vector<float> dW0,db0,dW1,db1,dW2,db2;
    std::vector<float> mW0,vW0,mb0,vb0;
    std::vector<float> mW1,vW1,mb1,vb1;
    std::vector<float> mW2,vW2,mb2,vb2;
    int adamT = 0;

    // Stored activations for single-sample backward
    std::vector<float> sa0;  // input copy
    std::vector<float> pa1;  // post-tanh h1
    std::vector<float> pa2;  // post-tanh h2

    MLP() : d0(0),d1(0),d2(0),d3(0) {}

    MLP(int in, int h1, int h2, int out) : d0(in), d1(h1), d2(h2), d3(out)
    {
        std::mt19937 rng(std::random_device{}());
        auto init = [&](std::vector<float>& w, int n, int m) {
            w.resize(n * m);
            float s = sqrtf(2.0f / n);
            std::normal_distribution<float> dist(0.0f, s);
            for (auto& x : w) x = dist(rng);
        };
        init(W0,d0,d1); b0.assign(d1,0.0f);
        init(W1,d1,d2); b1.assign(d2,0.0f);
        init(W2,d2,d3); b2.assign(d3,0.0f);

        dW0.assign(d0*d1,0); db0.assign(d1,0);
        dW1.assign(d1*d2,0); db1.assign(d2,0);
        dW2.assign(d2*d3,0); db2.assign(d3,0);

        mW0.assign(d0*d1,0); vW0.assign(d0*d1,0); mb0.assign(d1,0); vb0.assign(d1,0);
        mW1.assign(d1*d2,0); vW1.assign(d1*d2,0); mb1.assign(d2,0); vb1.assign(d2,0);
        mW2.assign(d2*d3,0); vW2.assign(d2*d3,0); mb2.assign(d3,0); vb2.assign(d3,0);

        sa0.resize(d0); pa1.resize(d1); pa2.resize(d2);
    }

    // Forward: x[d0] → out[d3], last layer linear, stores activations for backward
    void forward(const float* x, float* out)
    {
        for (int i = 0; i < d0; i++) sa0[i] = x[i];

        // Layer 0: pa1 = tanh(W0*x + b0)
        for (int j = 0; j < d1; j++) {
            float s = b0[j];
            for (int i = 0; i < d0; i++) s += W0[i*d1+j] * x[i];
            pa1[j] = tanhf(s);
        }
        // Layer 1: pa2 = tanh(W1*pa1 + b1)
        for (int j = 0; j < d2; j++) {
            float s = b1[j];
            for (int i = 0; i < d1; i++) s += W1[i*d2+j] * pa1[i];
            pa2[j] = tanhf(s);
        }
        // Layer 2: out = W2*pa2 + b2 (linear)
        for (int j = 0; j < d3; j++) {
            float s = b2[j];
            for (int i = 0; i < d2; i++) s += W2[i*d3+j] * pa2[i];
            out[j] = s;
        }
    }

    void zeroGrad()
    {
        auto z = [](std::vector<float>& v){ memset(v.data(),0,v.size()*sizeof(float)); };
        z(dW0);z(db0);z(dW1);z(db1);z(dW2);z(db2);
    }

    // Backward: accumulate gradients given dout[d3] for one sample
    void backward(const float* dout)
    {
        // Layer 2 → accumulate into dW2, db2; compute d_pa2
        std::vector<float> d_pa2(d2, 0.0f);
        for (int j = 0; j < d3; j++) {
            db2[j] += dout[j];
            for (int i = 0; i < d2; i++) {
                dW2[i*d3+j] += pa2[i] * dout[j];
                d_pa2[i]    += W2[i*d3+j] * dout[j];
            }
        }
        // Tanh' for layer 1: d_pre2 = d_pa2 * (1 - pa2^2)
        std::vector<float> d_pa1(d1, 0.0f);
        for (int j = 0; j < d2; j++) {
            float dp = d_pa2[j] * (1.0f - pa2[j]*pa2[j]);
            db1[j] += dp;
            for (int i = 0; i < d1; i++) {
                dW1[i*d2+j] += pa1[i] * dp;
                d_pa1[i]    += W1[i*d2+j] * dp;
            }
        }
        // Tanh' for layer 0
        for (int j = 0; j < d1; j++) {
            float dp = d_pa1[j] * (1.0f - pa1[j]*pa1[j]);
            db0[j] += dp;
            for (int i = 0; i < d0; i++)
                dW0[i*d1+j] += sa0[i] * dp;
        }
    }

    void applyAdam(float lr)
    {
        adamT++;
        float bc1 = 1.0f - powf(0.9f,   (float)adamT);
        float bc2 = 1.0f - powf(0.999f,  (float)adamT);

        auto step = [&](std::vector<float>& p, std::vector<float>& g,
                         std::vector<float>& m, std::vector<float>& v) {
            int n = (int)p.size();
            for (int i = 0; i < n; i++) {
                m[i] = 0.9f   * m[i] + 0.1f   * g[i];
                v[i] = 0.999f * v[i] + 0.001f * g[i]*g[i];
                p[i] -= lr * (m[i]/bc1) / (sqrtf(v[i]/bc2) + 1e-8f);
            }
        };
        step(W0,dW0,mW0,vW0); step(b0,db0,mb0,vb0);
        step(W1,dW1,mW1,vW1); step(b1,db1,mb1,vb1);
        step(W2,dW2,mW2,vW2); step(b2,db2,mb2,vb2);
    }
};

// ─────────── Rollout Buffer ───────────
struct RolloutBuffer
{
    float states   [ROLLOUT_LEN][STATE_DIM];
    float actions  [ROLLOUT_LEN][ACTION_DIM];
    float rewards  [ROLLOUT_LEN];
    float values   [ROLLOUT_LEN];
    float logprobs [ROLLOUT_LEN];
    float dones    [ROLLOUT_LEN];
    float advantages[ROLLOUT_LEN];
    float returns_ [ROLLOUT_LEN];

    int pos          = 0;
    int episodeCount = 0;

    RolloutBuffer() { clear(); }
    void clear() { pos = 0; episodeCount = 0; }
    bool isFull() const { return pos >= ROLLOUT_LEN; }

    void add(const float* state, const float* action, float reward,
             float value, float logprob, bool done)
    {
        if (pos >= ROLLOUT_LEN) return;
        memcpy(states[pos],  state,  STATE_DIM  * sizeof(float));
        memcpy(actions[pos], action, ACTION_DIM * sizeof(float));
        rewards[pos]  = reward;
        values[pos]   = value;
        logprobs[pos] = logprob;
        dones[pos]    = done ? 1.0f : 0.0f;
        if (done) episodeCount++;
        pos++;
    }

    void computeAdvantages(float lastValue)
    {
        float gae = 0.0f;
        for (int t = pos - 1; t >= 0; t--) {
            float nextV = (t == pos - 1) ? lastValue : values[t+1];
            float delta = rewards[t] + PPO_GAMMA * nextV * (1.0f - dones[t]) - values[t];
            gae = delta + PPO_GAMMA * PPO_LAMBDA * (1.0f - dones[t]) * gae;
            advantages[t] = gae;
            returns_[t]   = gae + values[t];
        }
        // Normalize advantages
        float mean = 0.0f;
        for (int t = 0; t < pos; t++) mean += advantages[t];
        mean /= pos;
        float var = 0.0f;
        for (int t = 0; t < pos; t++) {
            float d = advantages[t] - mean;
            var += d*d;
        }
        float invstd = 1.0f / sqrtf(var / pos + 1e-8f);
        for (int t = 0; t < pos; t++)
            advantages[t] = (advantages[t] - mean) * invstd;
    }
};

// ─────────── PPO Agent (split-head actor + critic) ───────────
struct PPOAgent
{
    MLP vaultTrunk; // VAULT_STATE_DIM → H1 → H2 → VAULT_ACTION_DIM*2 (means + log_stds)
    MLP setupHead;  // SETUP_STATE_DIM → SH1 → SH2 → 2 (mean + log_std for action[16])
    MLP critic;     // STATE_DIM → H1 → H2 → 1

    std::mt19937 rng;
    std::normal_distribution<float> ndist;

    PPOAgent()
        : vaultTrunk(VAULT_STATE_DIM, H1, H2, VAULT_ACTION_DIM * 2),
          setupHead (SETUP_STATE_DIM, SH1, SH2, 2),
          critic    (STATE_DIM, H1, H2, 1),
          rng(42u),
          ndist(0.0f, 1.0f)
    {}

    // Clamp log_std to [-2, 2], with pass-through gradient indicator
    static float clampLs(float x) { return x < -2.0f ? -2.0f : (x > 2.0f ? 2.0f : x); }
    static float clampLsGrad(float x) { return (x >= -2.0f && x <= 2.0f) ? 1.0f : 0.0f; }

    // Sample all 17 actions and compute log_prob + value
    void sample(const float* state, float* actions, float& logprob, float& value)
    {
        float vout[VAULT_ACTION_DIM * 2];
        vaultTrunk.forward(state, vout);

        float sout[2];
        setupHead.forward(state + VAULT_STATE_DIM, sout);

        logprob = 0.0f;

        // Vault actions [0..15]: plain Gaussian sample
        for (int i = 0; i < VAULT_ACTION_DIM; i++) {
            float mu  = vout[i];
            float ls  = clampLs(vout[VAULT_ACTION_DIM + i]);
            float sig = expf(ls);
            float z   = ndist(rng);
            actions[i] = mu + sig * z;
            logprob   += -0.5f * z * z - ls - 0.5f * LOG_2PI;
        }

        // Setup action [16]
        {
            float mu  = sout[0];
            float ls  = clampLs(sout[1]);
            float sig = expf(ls);
            float z   = ndist(rng);
            actions[ACTION_DIM - 1] = mu + sig * z;
            logprob += -0.5f * z * z - ls - 0.5f * LOG_2PI;
        }

        // Critic value
        float cval[1];
        critic.forward(state, cval);
        value = cval[0];
    }

    float getValue(const float* state)
    {
        float cval[1];
        critic.forward(state, cval);
        return cval[0];
    }

    void update(RolloutBuffer& buf)
    {
        int N = buf.pos;
        if (N < BATCH_SIZE) return;

        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);

        for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
            std::shuffle(idx.begin(), idx.end(), rng);

            for (int b = 0; b + BATCH_SIZE <= N; b += BATCH_SIZE) {
                vaultTrunk.zeroGrad();
                setupHead.zeroGrad();
                critic.zeroGrad();

                for (int bi = 0; bi < BATCH_SIZE; bi++) {
                    int t = idx[b + bi];
                    const float* s      = buf.states[t];
                    const float* a      = buf.actions[t];
                    float old_lp        = buf.logprobs[t];
                    float adv           = buf.advantages[t];
                    float ret           = buf.returns_[t];

                    // ── Actor forward ──
                    float vout[VAULT_ACTION_DIM * 2];
                    vaultTrunk.forward(s, vout);
                    float sout[2];
                    setupHead.forward(s + VAULT_STATE_DIM, sout);

                    // ── Compute new log_prob ──
                    float new_lp = 0.0f;
                    for (int i = 0; i < VAULT_ACTION_DIM; i++) {
                        float mu  = vout[i];
                        float ls  = clampLs(vout[VAULT_ACTION_DIM + i]);
                        float sig = expf(ls);
                        float res = (a[i] - mu) / sig;
                        new_lp   += -0.5f * res * res - ls - 0.5f * LOG_2PI;
                    }
                    {
                        float mu  = sout[0];
                        float ls  = clampLs(sout[1]);
                        float sig = expf(ls);
                        float res = (a[ACTION_DIM-1] - mu) / sig;
                        new_lp   += -0.5f * res * res - ls - 0.5f * LOG_2PI;
                    }

                    // ── PPO clip gradient w.r.t. new_lp ──
                    float r        = expf(new_lp - old_lp);
                    float r_clipped = r < (1.0f - PPO_CLIP_EPS) ? (1.0f - PPO_CLIP_EPS)
                                    : r > (1.0f + PPO_CLIP_EPS) ? (1.0f + PPO_CLIP_EPS) : r;
                    // clipped = (r > 1+eps && adv > 0) || (r < 1-eps && adv < 0)
                    bool clipped   = (r * adv > r_clipped * adv);
                    // d(loss=-L)/d(new_lp): negative gradient (we want to maximize L)
                    float d_lp = clipped ? 0.0f : (-adv * r) / BATCH_SIZE;

                    // ── Backprop log_prob → vout, sout ──
                    float d_vout[VAULT_ACTION_DIM * 2] = {};
                    float d_sout[2] = {};

                    for (int i = 0; i < VAULT_ACTION_DIM; i++) {
                        float mu  = vout[i];
                        float raw = vout[VAULT_ACTION_DIM + i];
                        float ls  = clampLs(raw);
                        float sig = expf(ls);
                        float ai  = a[i];
                        float res = ai - mu;

                        // d(new_lp)/d(mu) = (ai-mu)/sig^2
                        d_vout[i] += d_lp * (res / (sig * sig));

                        // d(new_lp)/d(ls) = (ai-mu)^2/sig^2 - 1
                        // entropy: d(-COEF*H)/d(ls) = -COEF (H=ls+const)
                        float d_ls = d_lp * (res * res / (sig * sig) - 1.0f)
                                   - PPO_ENTROPY_COEF / BATCH_SIZE;
                        d_vout[VAULT_ACTION_DIM + i] += d_ls * clampLsGrad(raw);
                    }
                    {
                        float mu  = sout[0];
                        float raw = sout[1];
                        float ls  = clampLs(raw);
                        float sig = expf(ls);
                        float ai  = a[ACTION_DIM - 1];
                        float res = ai - mu;
                        d_sout[0] += d_lp * (res / (sig * sig));
                        float d_ls = d_lp * (res * res / (sig * sig) - 1.0f)
                                   - PPO_ENTROPY_COEF / BATCH_SIZE;
                        d_sout[1] += d_ls * clampLsGrad(raw);
                    }

                    vaultTrunk.backward(d_vout);
                    setupHead.backward(d_sout);

                    // ── Critic forward + value loss ──
                    float cval[1];
                    critic.forward(s, cval);
                    float d_cval[1] = { PPO_VALUE_COEF * (cval[0] - ret) / BATCH_SIZE };
                    critic.backward(d_cval);
                }

                vaultTrunk.applyAdam(PPO_LR);
                setupHead.applyAdam(PPO_LR);
                critic.applyAdam(PPO_LR);
            }
        }
    }
};
