#include "Evaluation.h"
#include <cassert>
#include <limits>
int main(int argc,char**argv) {
    using namespace ovpp;
    MotionCache m;m.K=3;m.B=24;m.D=69;m.control_dt=.5;m.length=1;
    m.body_vel.assign(3*24*3,0);m.body_ang_vel.assign(3*24*3,0);m.dof_vel.assign(3*69,0);
    m.body_pos.assign(3*24*3,0);m.body_rot.assign(3*24*4,0);
    for(int i=0;i<3*24;++i)m.body_rot[i*4+3]=1;
    ArticulationState s;s.body_pos.assign(24,{.1f,0,0});s.body_rot.assign(24,{0,0,0,1});
    Evaluation e(.5);e.add(s,m,1,.5);e.add(s,m,2,1);
    std::string root=argv[1];e.write(root+"/pass.json",m,"quote\"\\\nmodel","cache",1);
    e.write(root+"/incomplete.json",m,"model","cache",.5);
    s.body_pos.assign(24,{.6f,0,0});e.add(s,m,2,1);
    e.write(root+"/fail.json",m,"model","cache",1);
    bool rejected=false;s.body_pos[0][0]=std::numeric_limits<float>::quiet_NaN();
    try{e.add(s,m,2,1);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
    rejected=false;try{Evaluation bad(std::numeric_limits<double>::quiet_NaN());}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
}
