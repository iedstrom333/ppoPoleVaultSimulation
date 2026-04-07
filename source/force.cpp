/*
* Copyright (c) 2026 Chris Giles
*
* Permission to use, copy, modify, distribute and sell this software
* and its documentation for any purpose is hereby granted without fee,
* provided that the above copyright notice appear in all copies.
* Chris Giles makes no representations about the suitability
* of this software for any purpose.
* It is provided "as is" without express or implied warranty.
*/

#include "solver.h"

Force::Force(Solver* solver, Rigid* bodyA, Rigid* bodyB)
    : solver(solver), bodyA(bodyA), bodyB(bodyB), nextA(0), nextB(0)
{
    // Add to solver linked list
    next = solver->forces;
    solver->forces = this;

    // Add to body linked lists
    if (bodyA)
    {
        nextA = bodyA->forces;
        bodyA->forces = this;
    }
    if (bodyB)
    {
        nextB = bodyB->forces;
        bodyB->forces = this;
    }
}


Force::~Force()
{
    // Remove from solver linked list
    Force** p = &solver->forces;
    while (*p != this)
        p = &(*p)->next;
    *p = next;

    // Remove from body linked lists
    if (bodyA)
    {
        p = &bodyA->forces;
        while (*p != this)
            p = (*p)->bodyA == bodyA ? &(*p)->nextA : &(*p)->nextB;
        *p = nextA;
    }

    if (bodyB)
    {
        p = &bodyB->forces;
        while (*p != this)
            p = (*p)->bodyA == bodyB ? &(*p)->nextA : &(*p)->nextB;
        *p = nextB;
    }
}

// ─────────── JointLimit ───────────

JointLimit::JointLimit(Solver* solver, Rigid* bodyA, Rigid* bodyB,
                       float3 axisA, float minAngle, float maxAngle, float stiffness)
    : Force(solver, bodyA, bodyB), axisA(axisA), minAngle(minAngle), maxAngle(maxAngle), stiffness(stiffness)
{
    // Store rest relative orientation so limits are measured as deltas from initial pose
    restRel = bodyB->positionAng * inverse(bodyA->positionAng);
}

float JointLimit::getCurrentAngle() const
{
    // Measure angle RELATIVE TO REST POSE so limits are offsets from initial configuration.
    // rel = current relative rotation * inverse(rest relative rotation)
    quat curRel  = bodyB->positionAng * inverse(bodyA->positionAng);
    quat delta   = curRel * inverse(restRel);
    float3 axisW = rotate(bodyA->positionAng, axisA);
    float3 proj  = axisW * dot(float3{delta.x, delta.y, delta.z}, axisW);
    quat twist   = normalize(quat{proj.x, proj.y, proj.z, delta.w});
    return 2.0f * atan2f(length(twist.vec()), twist.w) * sign(dot(twist.vec(), axisW));
}

void JointLimit::updatePrimal(Rigid* body, float /*alpha*/, float3x3& /*lhsLin*/,
                               float3x3& lhsAng, float3x3& /*lhsCross*/,
                               float3& /*rhsLin*/, float3& rhsAng)
{
    float angle     = getCurrentAngle();
    float violation = 0.0f;
    if      (angle > maxAngle) violation = angle - maxAngle;
    else if (angle < minAngle) violation = angle - minAngle;
    else return; // within limits — no contribution

    float3   axisW = rotate(bodyA->positionAng, axisA);
    float3x3 Kaa   = outer(axisW, axisW) * stiffness;
    float3   tau   = axisW * (stiffness * violation);

    lhsAng += Kaa;
    if (body == bodyA) rhsAng += tau;  // push parent opposite direction
    else               rhsAng -= tau;  // push child back into range
}
