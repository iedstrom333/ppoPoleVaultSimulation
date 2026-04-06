#pragma once
#include "solver.h"
#include "ppo.h"  // for MAX_EPISODE_STEPS

#ifdef __APPLE__
#include <OpenGL/GL.h>
#else
#include <GL/gl.h>
#endif

// ─────────── Athlete Profile ───────────
struct AthleteProfile
{
    float heightM         = 1.80f;
    float weightKg        = 75.0f;
    float takeoffSpeedMs  = 8.5f;
    float takeoffAngleDeg = 13.0f;

    // Strength (1-rep max in kg)
    float benchPressKg    = 80.0f;
    float squatKg         = 100.0f;
    float hangLegLiftKg   = 30.0f;

    // Pole
    float poleLengthM     = 5.0f;
    int   poleFlexRating  = 10;  // 0=stiffer, higher=more flexible

    float sizeScale()       const { return heightM / 1.80f; }
    float poleStiffness()   const { return fmaxf(1600.f, 40000.f - poleFlexRating * 2400.f); }
    float segLen()          const { return poleLengthM / 8.0f; }
    float armTorqueScale()  const { return benchPressKg  / 80.0f; }
    float legTorqueScale()  const { return squatKg        / 100.0f; }
    float coreTorqueScale() const { return hangLegLiftKg  / 30.0f; }
    float3 takeoffVelocity() const {
        float r = takeoffAngleDeg * 0.01745329f;
        return { takeoffSpeedMs * cosf(r), 0.0f, takeoffSpeedMs * sinf(r) };
    }
};

// ─────────── Competition State ───────────
struct CompetitionState
{
    float barHeight       = 0.0f;
    float minHeight       = 0.0f;
    float increment       = 0.15f;
    int   attemptsAtHeight= 0;
    int   totalClears     = 0;
    int   totalAttempts   = 0;
    float bestHeight      = 0.0f;

    enum class Outcome { Cleared, BarHit, GroundHit, Timeout };

    void update(Outcome outcome) {
        totalAttempts++;
        if (outcome == Outcome::Cleared) {
            totalClears++;
            bestHeight = bestHeight > barHeight ? bestHeight : barHeight;
            barHeight += increment;
            attemptsAtHeight = 0;
        } else {
            attemptsAtHeight++;
            if (attemptsAtHeight >= 3) {
                barHeight = minHeight;
                attemptsAtHeight = 0;
            }
        }
    }
};

// ─────────── Force History (per-attempt line graphs) ───────────
static const int FORCE_HIST_LEN = 500;

struct ForceHistory
{
    float data[16][FORCE_HIST_LEN];
    int   len = 0;

    void clear() { len = 0; }
    void record(const float* actions) {
        if (len >= FORCE_HIST_LEN) return;
        for (int i = 0; i < 16; i++) data[i][len] = fabsf(actions[i]);
        len++;
    }
};

// ─────────── Pole Vault Scene ───────────
struct PoleVaultScene
{
    AthleteProfile profile;
    Solver*        solver = nullptr;

    // Scene static bodies
    Rigid* groundBody  = nullptr;
    Rigid* groundAnchor= nullptr;
    Rigid* boxStopBoard= nullptr;
    Rigid* boxBottom   = nullptr;
    Rigid* boxWallL    = nullptr;
    Rigid* boxWallR    = nullptr;
    Rigid* uprightL    = nullptr;
    Rigid* uprightR    = nullptr;
    Rigid* crossbar    = nullptr;

    // Ragdoll
    Rigid* hips       = nullptr;
    Rigid* spine      = nullptr;
    Rigid* chest      = nullptr;
    Rigid* neck       = nullptr;
    Rigid* head       = nullptr;
    Rigid* lUpperArm  = nullptr;
    Rigid* lForearm   = nullptr;
    Rigid* lHand      = nullptr;
    Rigid* rUpperArm  = nullptr;
    Rigid* rForearm   = nullptr;
    Rigid* rHand      = nullptr;
    Rigid* lUpperLeg  = nullptr;
    Rigid* lLowerLeg  = nullptr;
    Rigid* lFoot      = nullptr;
    Rigid* rUpperLeg  = nullptr;
    Rigid* rLowerLeg  = nullptr;
    Rigid* rFoot      = nullptr;

    // Ordered persistent array set in buildRagdoll()
    // {hips,spine,chest,neck,head,
    //  lUpperArm,lForearm,lHand,
    //  rUpperArm,rForearm,rHand,
    //  lUpperLeg,lLowerLeg,lFoot,
    //  rUpperLeg,rLowerLeg,rFoot}
    Rigid* parts[17] = {};

    // Pole
    Rigid*  poleSegs[8] = {};
    Joint*  gripR = nullptr;
    Joint*  gripL = nullptr;

    // Competition-synced state (set from main before each step)
    float crossbarHeight   = 0.0f;
    int   attemptsAtHeight = 0;
    float bestHeight       = 0.0f;

    // Per-episode trackers
    int   stepCount              = 0;
    bool  releasedR              = false;
    bool  releasedL              = false;
    bool  barTouchedThisAttempt  = false;
    bool  wasAboveBar            = false;
    float peakHipsZ              = 0.0f;
    float standardsX             = 0.40f;

    // Carry-over from previous attempt
    float prevStandardsX  = 0.40f;
    float prevOutcome     = 1.0f;  // Timeout
    float prevPeakHipsZ   = 0.0f;
    bool  prevWasAboveBar = false;

    // ── Public interface ──────────────────────────────────────────────
    void setup(Solver* s) {
        solver = s;
        solver->clear();
        buildGround();
        buildBox();
        buildStandards();
        buildRagdoll();
        buildPole();
        suppressCollisions();
    }

    void reset() {
        // Save per-attempt state before clearing
        prevStandardsX  = standardsX;
        prevPeakHipsZ   = peakHipsZ;
        prevWasAboveBar = wasAboveBar;
        // prevOutcome is set by checkTerminal before reset is called

        // Reset current-episode trackers
        stepCount             = 0;
        releasedR             = false;
        releasedL             = false;
        barTouchedThisAttempt = false;
        wasAboveBar           = false;
        peakHipsZ             = 0.0f;
        gripR                 = nullptr;
        gripL                 = nullptr;

        solver->clear();
        buildGround();
        buildBox();
        buildStandards();
        buildRagdoll();
        buildPole();
        suppressCollisions();
    }

    void getState(float* out) const
    {
        quat hq     = hips->positionAng;
        quat hq_inv = { -hq.x, -hq.y, -hq.z, hq.w };  // conjugate

        // [0..2]: hips world position
        out[0] = hips->positionLin.x;
        out[1] = hips->positionLin.y;
        out[2] = hips->positionLin.z;

        // [3..5]: hips world linear velocity
        out[3] = hips->velocityLin.x;
        out[4] = hips->velocityLin.y;
        out[5] = hips->velocityLin.z;

        // [6..8]: hips orientation (quat x,y,z)
        out[6] = hq.x; out[7] = hq.y; out[8] = hq.z;

        // [9..11]: hips angular velocity
        out[9]  = hips->velocityAng.x;
        out[10] = hips->velocityAng.y;
        out[11] = hips->velocityAng.z;

        // [12..155]: 16 other bodies × 9 (relPos, relOrient, relVelLin — hips-local)
        int off = 12;
        for (int i = 1; i < 17; i++) {
            Rigid* b = parts[i];
            // relPos in hips-local frame
            float3 dp = { b->positionLin.x - hips->positionLin.x,
                          b->positionLin.y - hips->positionLin.y,
                          b->positionLin.z - hips->positionLin.z };
            float3 rp = rotate(hq_inv, dp);
            out[off+0] = rp.x; out[off+1] = rp.y; out[off+2] = rp.z;

            // relOrient: quat difference encoded as float3 (maths.h quat-quat op)
            float3 ro = b->positionAng - hq;
            out[off+3] = ro.x; out[off+4] = ro.y; out[off+5] = ro.z;

            // relVelLin in hips-local frame
            float3 dv = { b->velocityLin.x - hips->velocityLin.x,
                          b->velocityLin.y - hips->velocityLin.y,
                          b->velocityLin.z - hips->velocityLin.z };
            float3 rv = rotate(hq_inv, dv);
            out[off+6] = rv.x; out[off+7] = rv.y; out[off+8] = rv.z;
            off += 9;
        }
        // off == 156

        // [156..163]: pole segment bend angles (dot product of adjacent long-axes)
        for (int i = 0; i < 7; i++) {
            float3 axA = rotate(poleSegs[i]->positionAng,   float3{1,0,0});
            float3 axB = rotate(poleSegs[i+1]->positionAng, float3{1,0,0});
            out[156+i] = dot(axA, axB);
        }
        out[163] = 1.0f;  // seg[7] has no neighbor

        // [164..171]: pole segment angular velocity magnitudes
        for (int i = 0; i < 8; i++)
            out[164+i] = length(poleSegs[i]->velocityAng);

        // [172]: normalized episode time
        out[172] = (float)stepCount / MAX_EPISODE_STEPS;

        // [173]: current standards X, normalized (/0.80)
        out[173] = standardsX / 0.80f;

        // [174]: bar height, normalized (/6.0)
        out[174] = crossbarHeight / 6.0f;

        // [175]: prev standards X, normalized
        out[175] = prevStandardsX / 0.80f;

        // [176]: prev outcome (Cleared=0, BarHit=0.33, GroundHit=0.67, Timeout=1.0)
        out[176] = prevOutcome;

        // [177]: prev peak hips Z, normalized vs crossbar height
        float normH = crossbarHeight > 0.1f ? crossbarHeight : 3.0f;
        out[177] = prevPeakHipsZ / normH;

        // [178]: attempts used at current height, normalized (/3.0)
        out[178] = (float)attemptsAtHeight / 3.0f;

        // [179]: personal best height, normalized (/6.0)
        out[179] = bestHeight / 6.0f;

        // [180]: was above bar on previous attempt
        out[180] = prevWasAboveBar ? 1.0f : 0.0f;
    }

    void applyActions(const float* actions, float dt)
    {
        float s = profile.sizeScale();
        float arm  = profile.armTorqueScale();
        float leg  = profile.legTorqueScale();
        float core = profile.coreTorqueScale();

        // Torque application helper (equal-and-opposite)
        auto torque = [&](Rigid* parent, Rigid* child, float3 worldTorque) {
            applyJointTorque(parent, worldTorque, dt);
            applyJointTorque(child, {-worldTorque.x,-worldTorque.y,-worldTorque.z}, dt);
        };

        // [0]: Spine flex/ext (Y axis of hips frame)
        float3 spineY = rotate(hips->positionAng, float3{0,1,0});
        torque(hips, spine, spineY * (actions[0] * 29.7f * core));

        // [1]: Chest rotation (Y axis of spine frame)
        float3 chestY = rotate(spine->positionAng, float3{0,1,0});
        torque(spine, chest, chestY * (actions[1] * 19.8f * core));

        // [2]: L Shoulder pitch (Y)
        float3 lShY = rotate(chest->positionAng, float3{0,1,0});
        torque(chest, lUpperArm, lShY * (actions[2] * 19.8f * arm));

        // [3]: L Shoulder yaw (Z)
        float3 lShZ = rotate(chest->positionAng, float3{0,0,1});
        torque(chest, lUpperArm, lShZ * (actions[3] * 14.9f * arm));

        // [4]: L Elbow flex (Y)
        float3 lElY = rotate(lUpperArm->positionAng, float3{0,1,0});
        torque(lUpperArm, lForearm, lElY * (actions[4] * 14.9f * arm));

        // [5]: R Shoulder pitch (Y)
        float3 rShY = rotate(chest->positionAng, float3{0,1,0});
        torque(chest, rUpperArm, rShY * (actions[5] * 19.8f * arm));

        // [6]: R Shoulder yaw (Z)
        float3 rShZ = rotate(chest->positionAng, float3{0,0,1});
        torque(chest, rUpperArm, rShZ * (actions[6] * 14.9f * arm));

        // [7]: R Elbow flex (Y)
        float3 rElY = rotate(rUpperArm->positionAng, float3{0,1,0});
        torque(rUpperArm, rForearm, rElY * (actions[7] * 14.9f * arm));

        // [8]: L Hip flex (Y)
        float3 lHipY = rotate(hips->positionAng, float3{0,1,0});
        torque(hips, lUpperLeg, lHipY * (actions[8] * 29.7f * leg));

        // [9]: L Hip abduction (Z)
        float3 lHipZ = rotate(hips->positionAng, float3{0,0,1});
        torque(hips, lUpperLeg, lHipZ * (actions[9] * 14.9f * leg));

        // [10]: L Knee flex (Y)
        float3 lKnY = rotate(lUpperLeg->positionAng, float3{0,1,0});
        torque(lUpperLeg, lLowerLeg, lKnY * (actions[10] * 19.8f * leg));

        // [11]: R Hip flex (Y)
        float3 rHipY = rotate(hips->positionAng, float3{0,1,0});
        torque(hips, rUpperLeg, rHipY * (actions[11] * 29.7f * leg));

        // [12]: R Hip abduction (Z)
        float3 rHipZ = rotate(hips->positionAng, float3{0,0,1});
        torque(hips, rUpperLeg, rHipZ * (actions[12] * 14.9f * leg));

        // [13]: R Knee flex (Y)
        float3 rKnY = rotate(rUpperLeg->positionAng, float3{0,1,0});
        torque(rUpperLeg, rLowerLeg, rKnY * (actions[13] * 19.8f * leg));

        // [14]: Release R grip (latched)
        if (!releasedR && actions[14] > 0.8f) {
            delete gripR;
            gripR    = nullptr;
            releasedR = true;
        }
        // [15]: Release L grip (latched)
        if (!releasedL && actions[15] > 0.8f) {
            delete gripL;
            gripL    = nullptr;
            releasedL = true;
        }

        stepCount++;
    }

    void setStandardsPosition(float x)
    {
        standardsX = x;
        float h = crossbarHeight;
        if (uprightL) {
            uprightL->positionLin = {x,  2.25f, h * 0.5f};
            uprightR->positionLin = {x, -2.25f, h * 0.5f};
            crossbar->positionLin = {x,  0.0f,  h};
            uprightL->size = {0.1f, 0.1f, h + 0.5f};
            uprightR->size = {0.1f, 0.1f, h + 0.5f};
        }
    }

    struct Result { float reward; bool done; CompetitionState::Outcome outcome; };

    Result checkTerminal()
    {
        float3 hPos = hips->positionLin;

        // Update peak height flag
        if (hPos.x > -0.5f && hPos.z > crossbarHeight + 0.05f)
            wasAboveBar = true;
        if (hPos.z > peakHipsZ) peakHipsZ = hPos.z;

        // Bar touch check — bounding-sphere vs AABB of static crossbar
        if (!barTouchedThisAttempt) {
            float3 cbPos  = crossbar->positionLin;
            float3 cbHalf = { crossbar->size.x * 0.5f,
                              crossbar->size.y * 0.5f,
                              crossbar->size.z * 0.5f };
            auto touches = [&](Rigid* b) -> bool {
                float3 d  = { b->positionLin.x - cbPos.x,
                              b->positionLin.y - cbPos.y,
                              b->positionLin.z - cbPos.z };
                float r   = b->radius;
                float dx  = fmaxf(0.0f, fabsf(d.x) - cbHalf.x);
                float dy  = fmaxf(0.0f, fabsf(d.y) - cbHalf.y);
                float dz  = fmaxf(0.0f, fabsf(d.z) - cbHalf.z);
                return dx*dx + dy*dy + dz*dz <= r*r;
            };
            for (int i = 0; i < 17 && !barTouchedThisAttempt; i++)
                if (touches(parts[i])) barTouchedThisAttempt = true;
            for (int j = 0; j < 8 && !barTouchedThisAttempt; j++)
                if (touches(poleSegs[j])) barTouchedThisAttempt = true;
        }

        // ── Terminal conditions ──

        // Cleared: was above bar, bar not touched, now descending
        if (wasAboveBar && !barTouchedThisAttempt && hPos.z < crossbarHeight) {
            prevOutcome = 0.0f;
            return { 100.0f, true, CompetitionState::Outcome::Cleared };
        }

        // Bar touched
        if (barTouchedThisAttempt) {
            prevOutcome = 0.33f;
            return { -50.0f, true, CompetitionState::Outcome::BarHit };
        }

        // Fell to ground on runway side
        if (hPos.z < 0.10f && hPos.x < 0.0f) {
            prevOutcome = 0.67f;
            return { -20.0f, true, CompetitionState::Outcome::GroundHit };
        }

        // Timeout
        if (stepCount >= MAX_EPISODE_STEPS) {
            prevOutcome = 1.0f;
            return { 0.0f, true, CompetitionState::Outcome::Timeout };
        }

        // Out of bounds — too far from runway or sideways
        if (fabsf(hPos.y) > 5.0f || hPos.x < -20.0f || hPos.x > 10.0f || hPos.z > 20.0f) {
            prevOutcome = 0.67f;
            return { -50.0f, true, CompetitionState::Outcome::GroundHit };
        }

        // Living penalty + shaping: reward forward progress and height
        float shapingReward = -0.05f;
        if (hPos.z > 1.0f) shapingReward += 0.01f * hPos.z;  // reward getting up
        return { shapingReward, false, CompetitionState::Outcome::Timeout };
    }

    void drawExtras()
    {
        // Draw crossbar height reference lines
        glColor3f(1.0f, 0.5f, 0.0f);
        glBegin(GL_LINES);
        // Horizontal bar height indicator
        glVertex3f(-5.0f, 0.0f, crossbarHeight);
        glVertex3f( 5.0f, 0.0f, crossbarHeight);
        glEnd();
    }

private:
    // ── Apply torque as pre-step velocity impulse ──
    void applyJointTorque(Rigid* body, float3 worldTorque, float dt)
    {
        if (!body || body->mass == 0.0f) return;
        float3 tLocal = rotate({ -body->positionAng.x, -body->positionAng.y,
                                  -body->positionAng.z,  body->positionAng.w }, worldTorque);
        float3 angAcc = {
            tLocal.x / body->moment.x,
            tLocal.y / body->moment.y,
            tLocal.z / body->moment.z
        };
        float3 angAccW = rotate(body->positionAng, angAcc);
        body->velocityAng.x += angAccW.x * dt;
        body->velocityAng.y += angAccW.y * dt;
        body->velocityAng.z += angAccW.z * dt;

        // Velocity damping to prevent runaway motion
        float damp = 0.90f;
        body->velocityAng.x *= damp;
        body->velocityAng.y *= damp;
        body->velocityAng.z *= damp;
        body->velocityLin.x *= 0.98f;
        body->velocityLin.y *= 0.98f;
        body->velocityLin.z *= 0.98f;
    }

    // ── Build ground + anchor ──
    void buildGround()
    {
        groundBody   = new Rigid(solver, {30.0f, 10.0f, 0.5f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.25f});
        groundAnchor = new Rigid(solver, {0.05f, 0.05f, 0.05f}, 0.0f, 0.5f, {0.0f, 0.0f, 0.0f});
    }

    // ── Build IAAF plant box (visual-only, all collisions suppressed) ──
    void buildBox()
    {
        // Stop board: slight lean toward runway
        float angSB = rad(15.0f);
        boxStopBoard = new Rigid(solver, {0.04f, 0.20f, 0.30f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.12f});
        boxStopBoard->positionAng = {0.0f, sinf(angSB*0.5f), 0.0f, cosf(angSB*0.5f)};

        // Box bottom: slopes down to pit
        float angBot = -rad(9.3f);
        boxBottom = new Rigid(solver, {1.22f, 0.55f, 0.04f}, 0.0f, 0.5f, {-0.61f, 0.0f, -0.10f});
        boxBottom->positionAng = {0.0f, sinf(angBot*0.5f), 0.0f, cosf(angBot*0.5f)};

        // Left wall: taper inward
        float angWL = rad(30.0f);
        boxWallL = new Rigid(solver, {1.22f, 0.04f, 0.30f}, 0.0f, 0.5f, {-0.61f, -0.22f, -0.08f});
        boxWallL->positionAng = {sinf(angWL*0.5f), 0.0f, 0.0f, cosf(angWL*0.5f)};

        // Right wall: mirror
        float angWR = -rad(30.0f);
        boxWallR = new Rigid(solver, {1.22f, 0.04f, 0.30f}, 0.0f, 0.5f, {-0.61f,  0.22f, -0.08f});
        boxWallR->positionAng = {sinf(angWR*0.5f), 0.0f, 0.0f, cosf(angWR*0.5f)};
    }

    // ── Build standards and crossbar ──
    void buildStandards()
    {
        float h = crossbarHeight;
        float x = standardsX;
        uprightL = new Rigid(solver, {0.1f, 0.1f, h + 0.5f}, 0.0f, 0.5f, {x,  2.25f, h * 0.5f});
        uprightR = new Rigid(solver, {0.1f, 0.1f, h + 0.5f}, 0.0f, 0.5f, {x, -2.25f, h * 0.5f});
        crossbar = new Rigid(solver, {0.05f, 4.4f, 0.05f},   0.0f, 0.5f, {x,  0.0f,  h});
    }

    // ── Build 17-part ragdoll ──
    void buildRagdoll()
    {
        float s    = profile.sizeScale();
        float mscl = profile.weightKg / 75.0f;
        float3 v0  = profile.takeoffVelocity();

        auto makePart = [&](float3 sz, float massFrac, float3 pos) -> Rigid* {
            float3 ssz = sz * s;
            float density = massFrac * profile.weightKg / (ssz.x * ssz.y * ssz.z);
            Rigid* b = new Rigid(solver, ssz, density, 0.4f, pos * s);
            b->velocityLin = v0;
            return b;
        };

        float hx = -4.3f;

        // Positions derived from joint attachment points so gaps are zero at spawn.
        // Each body center = parent_joint_world + halfsize in Z.
        // hips center at (hx, 0, 1.02)
        hips      = makePart({0.35f,0.25f,0.20f}, 0.13f, {hx,  0.00f, 1.020f});
        // spine bottom at hips+{0,0,+0.10} = (hx,0,1.12), center = 1.12+0.125
        spine     = makePart({0.25f,0.20f,0.25f}, 0.11f, {hx,  0.00f, 1.245f});
        // chest bottom at spine+{0,0,+0.125} = (hx,0,1.37), center = 1.37+0.15
        chest     = makePart({0.36f,0.24f,0.30f}, 0.27f, {hx,  0.00f, 1.520f});
        // neck bottom at chest+{0,0,+0.15} = (hx,0,1.67), center = 1.67+0.06
        neck      = makePart({0.10f,0.10f,0.12f}, 0.01f, {hx,  0.00f, 1.730f});
        // head bottom at neck+{0,0,+0.06} = (hx,0,1.79), center = 1.79+0.12
        head      = makePart({0.22f,0.22f,0.24f}, 0.07f, {hx,  0.00f, 1.910f});

        // lUpperArm: chest attach at chest_center+{0,+0.12,+0.10} = (hx,0.12,1.62)
        // lUpperArm bottom rB={0,0,-0.15}, so center = attach + 0.15 - 0.15 = (hx,0.12,1.62)
        // Actually center = attach_world - rB_in_body = (hx,0.12,1.62)+(0,0,+0.15) = (hx,0.12,1.77) NO
        // rB={0,0,-0.15} means joint is at local -0.15z = bottom of arm
        // so world_joint=(hx,0.12,1.62), arm_center = world_joint - (0,0,-0.15) = (hx,0.12,1.77)
        lUpperArm = makePart({0.09f,0.09f,0.30f}, 0.03f, {hx,     0.12f, 1.770f});
        // lForearm: attach at lUpperArm_center+(0,0,+0.15)=(hx,0.12,1.92), rB={0,0,-0.135}
        // center = (hx,0.12,1.92)+(0,0,0.135) = (hx,0.12,2.055)
        lForearm  = makePart({0.07f,0.07f,0.27f}, 0.02f, {hx,     0.12f, 2.055f});
        // lHand: attach at lForearm_center+(0,0,+0.135)=(hx,0.12,2.19), rB={0,0,-0.05}
        // center = (hx,0.12,2.19)+(0,0,0.05) = (hx,0.12,2.24)
        lHand     = makePart({0.09f,0.05f,0.10f}, 0.007f,{hx,     0.12f, 2.240f});

        rUpperArm = makePart({0.09f,0.09f,0.30f}, 0.03f, {hx,    -0.12f, 1.770f});
        rForearm  = makePart({0.07f,0.07f,0.27f}, 0.02f, {hx,    -0.12f, 2.055f});
        rHand     = makePart({0.09f,0.05f,0.10f}, 0.007f,{hx,    -0.12f, 2.240f});

        // lUpperLeg: hips attach at hips_center+{0,+0.09,-0.10}=(hx,0.09,0.92)
        // rB={0,0,+0.215} means joint at local +0.215z = top of leg
        // center = (hx,0.09,0.92)-(0,0,0.215) = (hx,0.09,0.705)
        lUpperLeg = makePart({0.14f,0.14f,0.43f}, 0.11f, {hx,  0.09f, 0.705f});
        // lLowerLeg: attach at lUpperLeg_center+(0,0,-0.215)=(hx,0.09,0.49), rB={0,0,+0.20}
        // center = (hx,0.09,0.49)-(0,0,0.20) = (hx,0.09,0.29)
        lLowerLeg = makePart({0.09f,0.09f,0.40f}, 0.05f, {hx,  0.09f, 0.290f});
        // lFoot: attach at lLowerLeg_center+(0,0,-0.20)=(hx,0.09,0.09), rB={-0.05,0,+0.045}
        // center = (hx,0.09,0.09)-(-0.05,0,0.045) = (hx+0.05,0.09,0.045)
        lFoot     = makePart({0.25f,0.10f,0.09f}, 0.01f, {hx+0.05f, 0.09f, 0.045f});

        rUpperLeg = makePart({0.14f,0.14f,0.43f}, 0.11f, {hx, -0.09f, 0.705f});
        rLowerLeg = makePart({0.09f,0.09f,0.40f}, 0.05f, {hx, -0.09f, 0.290f});
        rFoot     = makePart({0.25f,0.10f,0.09f}, 0.01f, {hx+0.05f,-0.09f, 0.045f});

        // Populate parts array (index 0 = hips, then the 16 others)
        parts[0]  = hips;
        parts[1]  = spine;     parts[2]  = chest;    parts[3]  = neck;
        parts[4]  = head;
        parts[5]  = lUpperArm; parts[6]  = lForearm; parts[7]  = lHand;
        parts[8]  = rUpperArm; parts[9]  = rForearm; parts[10] = rHand;
        parts[11] = lUpperLeg; parts[12] = lLowerLeg;parts[13] = lFoot;
        parts[14] = rUpperLeg; parts[15] = rLowerLeg;parts[16] = rFoot;

        // ── Joints (hard pin, free rotation) ──
        auto J = [&](Rigid* a, Rigid* b, float3 rA, float3 rB) {
            new Joint(solver, a, b, rA*s, rB*s, 1e5f, 0.0f);
        };

        J(hips,      spine,    {0,0,+0.100f}, {0,0,-0.125f});
        J(spine,     chest,    {0,0,+0.125f}, {0,0,-0.150f});
        J(chest,     neck,     {0,0,+0.150f}, {0,0,-0.060f});
        J(neck,      head,     {0,0,+0.060f}, {0,0,-0.120f});
        J(chest,     lUpperArm,{0,+0.12f,+0.100f}, {0,0,-0.150f});
        J(lUpperArm, lForearm, {0,0,+0.150f}, {0,0,-0.135f});
        J(lForearm,  lHand,    {0,0,+0.135f}, {0,0,-0.050f});
        J(chest,     rUpperArm,{0,-0.12f,+0.100f},{0,0,-0.150f});
        J(rUpperArm, rForearm, {0,0,+0.150f}, {0,0,-0.135f});
        J(rForearm,  rHand,    {0,0,+0.135f}, {0,0,-0.050f});
        J(hips,      lUpperLeg,{0,+0.09f,-0.100f},{0,0,+0.215f});
        J(lUpperLeg, lLowerLeg,{0,0,-0.215f}, {0,0,+0.200f});
        J(lLowerLeg, lFoot,    {0,0,-0.200f}, {-0.05f,0,+0.045f});
        J(hips,      rUpperLeg,{0,-0.09f,-0.100f},{0,0,+0.215f});
        J(rUpperLeg, rLowerLeg,{0,0,-0.215f}, {0,0,+0.200f});
        J(rLowerLeg, rFoot,    {0,0,-0.200f}, {-0.05f,0,+0.045f});

        // ── Angular joint limits ──
        auto L = [&](Rigid* a, Rigid* b, float3 ax, float mn, float mx, float k=1e5f) {
            new JointLimit(solver, a, b, ax, rad(mn), rad(mx), k);
        };
        // Spine
        L(hips,spine,     {0,1,0}, -30.f, +80.f);
        L(hips,spine,     {1,0,0}, -30.f, +30.f);
        L(hips,spine,     {0,0,1}, -45.f, +45.f);
        // Chest
        L(spine,chest,    {0,1,0}, -15.f, +25.f);
        L(spine,chest,    {1,0,0}, -20.f, +20.f);
        L(spine,chest,    {0,0,1}, -30.f, +30.f);
        // Neck
        L(chest,neck,     {0,1,0}, -50.f, +50.f);
        L(chest,neck,     {1,0,0}, -45.f, +45.f);
        L(chest,neck,     {0,0,1}, -80.f, +80.f);
        // Head
        L(neck,head,      {0,1,0}, -10.f, +10.f);
        L(neck,head,      {1,0,0}, -10.f, +10.f);
        L(neck,head,      {0,0,1}, -10.f, +10.f);
        // Shoulders
        L(chest,lUpperArm,{0,1,0}, -60.f,+180.f);
        L(chest,lUpperArm,{1,0,0},   0.f,+180.f);
        L(chest,lUpperArm,{0,0,1}, -90.f, +90.f);
        L(chest,rUpperArm,{0,1,0}, -60.f,+180.f);
        L(chest,rUpperArm,{1,0,0},   0.f,+180.f);
        L(chest,rUpperArm,{0,0,1}, -90.f, +90.f);
        // Elbows (-5° buffer absorbs initial pose imprecision)
        L(lUpperArm,lForearm,{0,1,0},  -5.f,+145.f);
        L(rUpperArm,rForearm,{0,1,0},  -5.f,+145.f);
        // Wrists
        L(lForearm,lHand,  {0,1,0}, -70.f, +90.f);
        L(lForearm,lHand,  {1,0,0}, -25.f, +25.f);
        L(rForearm,rHand,  {0,1,0}, -70.f, +90.f);
        L(rForearm,rHand,  {1,0,0}, -25.f, +25.f);
        // Hips
        L(hips,lUpperLeg, {0,1,0}, -30.f,+120.f);
        L(hips,lUpperLeg, {1,0,0}, -20.f, +45.f);
        L(hips,lUpperLeg, {0,0,1}, -45.f, +45.f);
        L(hips,rUpperLeg, {0,1,0}, -30.f,+120.f);
        L(hips,rUpperLeg, {1,0,0}, -20.f, +45.f);
        L(hips,rUpperLeg, {0,0,1}, -45.f, +45.f);
        // Knees (-5° buffer)
        L(lUpperLeg,lLowerLeg,{0,1,0}, -5.f,+140.f);
        L(rUpperLeg,rLowerLeg,{0,1,0}, -5.f,+140.f);
        // Ankles
        L(lLowerLeg,lFoot, {0,1,0}, -20.f, +50.f);
        L(lLowerLeg,lFoot, {1,0,0}, -20.f, +20.f);
        L(rLowerLeg,rFoot, {0,1,0}, -20.f, +50.f);
        L(rLowerLeg,rFoot, {1,0,0}, -20.f, +20.f);
    }

    // ── Build pole (8 segments) + pin joint at tip ──
    void buildPole()
    {
        float s       = profile.sizeScale();
        float segLen  = profile.segLen();
        float halfSeg = segLen * 0.5f;
        float3 segSz  = { segLen, 0.04f * s, 0.04f * s };

        // Pole direction: from tip at (0,0,0) toward grip at athlete hips position
        float gripX = -4.3f * s;
        float gripZ =  1.50f * s;
        float len   = sqrtf(gripX*gripX + gripZ*gripZ);
        float3 poleDir = { gripX / len, 0.0f, gripZ / len };

        // Rotation of pole segments around Y axis
        float ang = atan2f(poleDir.z, -poleDir.x);
        quat segRot = { 0.0f, sinf(ang * 0.5f), 0.0f, cosf(ang * 0.5f) };

        float3 v0 = profile.takeoffVelocity();
        for (int i = 0; i < 8; i++) {
            float3 ctr = poleDir * ((halfSeg + i * segLen));
            poleSegs[i] = new Rigid(solver, segSz, 200.0f, 0.3f, ctr);
            poleSegs[i]->positionAng = segRot;
            poleSegs[i]->velocityLin = v0;
        }

        // Pin tip to ground anchor: INFINITY linear stiffness, 0 angular (free rotation)
        new Joint(solver, groundAnchor, poleSegs[0], {0,0,0}, {-halfSeg,0,0}, INFINITY, 0.0f);

        // Inter-segment bending stiffness
        float ks = profile.poleStiffness();
        for (int i = 0; i < 7; i++)
            new Joint(solver, poleSegs[i], poleSegs[i+1],
                      { halfSeg,0,0}, {-halfSeg,0,0}, 1e6f, ks);

        // Grip joints: right hand on seg[7] (upper grip), left hand on seg[6] (lower grip)
        gripR = new Joint(solver, rHand, poleSegs[7],
                          {0,0,+0.05f*s}, { halfSeg,0,0}, 1e6f, 0.0f);
        gripL = new Joint(solver, lHand, poleSegs[6],
                          {0,0,+0.05f*s}, {0,0,0},        1e6f, 0.0f);
    }

    // ── Suppress unwanted collisions ──
    void suppressCollisions()
    {
        // All non-adjacent ragdoll pairs
        for (int i = 0; i < 17; i++)
            for (int j = i+1; j < 17; j++)
                if (!parts[i]->constrainedTo(parts[j]))
                    new IgnoreCollision(solver, parts[i], parts[j]);

        // All ragdoll vs pole segments (excluding grip pairs)
        for (int i = 0; i < 17; i++)
            for (int j = 0; j < 8; j++)
                if (!parts[i]->constrainedTo(poleSegs[j]))
                    new IgnoreCollision(solver, parts[i], poleSegs[j]);

        // Non-adjacent pole segment pairs
        for (int i = 0; i < 8; i++)
            for (int j = i+2; j < 8; j++)
                new IgnoreCollision(solver, poleSegs[i], poleSegs[j]);

        // Plant box vs everything (visual only)
        Rigid* boxParts[4] = { boxStopBoard, boxBottom, boxWallL, boxWallR };
        for (int b = 0; b < 4; b++) {
            for (int i = 0; i < 17; i++)
                new IgnoreCollision(solver, parts[i], boxParts[b]);
            for (int j = 0; j < 8; j++)
                new IgnoreCollision(solver, poleSegs[j], boxParts[b]);
            for (int b2 = b+1; b2 < 4; b2++)
                new IgnoreCollision(solver, boxParts[b], boxParts[b2]);
            new IgnoreCollision(solver, groundBody, boxParts[b]);
            new IgnoreCollision(solver, groundAnchor, boxParts[b]);
        }

        // Ground anchor vs everything except poleSegs[0] (they share a joint)
        for (int i = 0; i < 17; i++)
            new IgnoreCollision(solver, groundAnchor, parts[i]);
        for (int j = 1; j < 8; j++)
            new IgnoreCollision(solver, groundAnchor, poleSegs[j]);

        // Standards / crossbar vs ground and box
        Rigid* standards[3] = { uprightL, uprightR, crossbar };
        for (int k = 0; k < 3; k++) {
            new IgnoreCollision(solver, standards[k], groundBody);
            new IgnoreCollision(solver, standards[k], groundAnchor);
            for (int b = 0; b < 4; b++)
                new IgnoreCollision(solver, standards[k], boxParts[b]);
            // Standards don't collide with each other
            for (int k2 = k+1; k2 < 3; k2++)
                new IgnoreCollision(solver, standards[k], standards[k2]);
            // Standards don't collide with pole or ragdoll (they're at Y=±2.25)
            for (int i = 0; i < 17; i++)
                new IgnoreCollision(solver, standards[k], parts[i]);
            for (int j = 0; j < 8; j++)
                new IgnoreCollision(solver, standards[k], poleSegs[j]);
        }
    }
};
