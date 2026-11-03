#pragma once
#include "SmplArticulation.h"
#include "MotionCache.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ovpp {
// Absolute world-space error; no reference reanchoring or early termination.
// A pass matches the campaign's 0.5 m maximum mean-body error criterion.
class Evaluation {
    std::vector<double> frame_error_, body_error_, rotation_error_;
    std::vector<std::vector<double>> limb_error_{5};
    double first_failure_ = -1;
    double threshold_;
    static std::string quote(const std::string& s) {
        std::string out="\"";
        for (unsigned char c : s) {
            if (c=='"' || c=='\\') {out+='\\';out+=c;}
            else if (c<32) { const char* h="0123456789abcdef"; out+="\\u00";out+=h[c>>4];out+=h[c&15]; }
            else out+=c;
        }
        return out+'"';
    }
    static double mean(const std::vector<double>& v) {
        return v.empty()?0:std::accumulate(v.begin(),v.end(),0.0)/v.size();
    }
    static double percentile(std::vector<double> v, double p) {
        if(v.empty()) return 0;
        std::sort(v.begin(),v.end());return v[(size_t)std::ceil(p*(v.size()-1))];
    }
public:
    explicit Evaluation(double threshold):threshold_(threshold) {
        if(!std::isfinite(threshold) || threshold<=0) throw std::runtime_error("Invalid evaluation threshold");
    }
    void add(const ArticulationState& s,const MotionCache& m,int tick,double time) {
        if(m.B!=24) throw std::runtime_error("Evaluation limb mapping requires SMPL 24 bodies");
        double sum=0; double limbs[5]={}; int counts[5]={};
        for(int b=0;b<m.B;++b) {
            double sq=0,dot=0,ns=0,nr=0;
            for(int j=0;j<3;++j) {double d=s.body_pos[b][j]-m.bodyPos(tick)[b*3+j];sq+=d*d;}
            for(int j=0;j<4;++j) {double x=s.body_rot[b][j],y=m.bodyRot(tick)[b*4+j];dot+=x*y;ns+=x*x;nr+=y*y;}
            double err=std::sqrt(sq);
            if(!std::isfinite(err) || !std::isfinite(dot) || ns<1e-12 || nr<1e-12)
                throw std::runtime_error("Non-finite simulation state or invalid rotation; evaluation invalid");
            double angle=2*std::acos(std::min(1.0,std::abs(dot)/std::sqrt(ns*nr)));
            body_error_.push_back(err);rotation_error_.push_back(angle);sum+=err;
            int limb=(b>=14&&b<=18)?0:(b>=19)?1:(b>=1&&b<=4)?2:(b>=5&&b<=8)?3:4;
            limbs[limb]+=err;++counts[limb];
        }
        double err=sum/m.B;frame_error_.push_back(err);
        for(int i=0;i<5;++i)limb_error_[i].push_back(limbs[i]/counts[i]);
        if(err>threshold_ && first_failure_<0)first_failure_=time;
    }
    void write(const std::string& path,const MotionCache& m,const std::string& model,
               const std::string& cache,double elapsed) const {
        bool complete=elapsed+1e-6>=m.length;
        bool passed=complete&&!frame_error_.empty()&&first_failure_<0;
        double peak_speed=0,peak_root=0,peak_ang=0,peak_joint=0,peak_accel=0;
        for(int t=0;t<std::min(m.K,(int)std::ceil(m.length/m.control_dt)+1);++t) {
            for(int b=0;b<m.B;++b) {
                double v=0,w=0,a=0;
                for(int j=0;j<3;++j) {
                    double x=m.bodyVel(t)[b*3+j],y=m.bodyAngVel(t)[b*3+j];v+=x*x;w+=y*y;
                    if(t>0){double d=(x-m.bodyVel(t-1)[b*3+j])/m.control_dt;a+=d*d;}
                }
                peak_speed=std::max(peak_speed,std::sqrt(v));peak_ang=std::max(peak_ang,std::sqrt(w));
                peak_accel=std::max(peak_accel,std::sqrt(a));if(b==0)peak_root=std::max(peak_root,std::sqrt(v));
            }
            for(int j=0;j<m.D;++j)peak_joint=std::max(peak_joint,(double)std::abs(m.dofVel(t)[j]));
        }
        std::filesystem::path dest(path),tmp(path+".tmp");
        if(dest.has_parent_path())std::filesystem::create_directories(dest.parent_path());
        std::ofstream f(tmp);if(!f)throw std::runtime_error("Cannot write evaluation report");
        f<<std::setprecision(9)<<"{\n  \"schema_version\": 2,\n  \"model\": "<<quote(model)
         <<",\n  \"motion_cache\": "<<quote(cache)<<",\n  \"complete\": "<<(complete?"true":"false")
         <<",\n  \"passed\": "<<(passed?"true":"false")
         <<",\n  \"score\": "<<100/(1.0+mean(frame_error_)/threshold_)
         <<",\n  \"score_definition\": \"100 / (1 + mean_position_error_m / failure_threshold_m); quality index, not probability\""
         <<",\n  \"assessment\": "<<quote(!complete?"Incomplete evaluation":passed?"Trackable by this model in standalone PhysX":"Tracking failure; not proof of physically impossible source motion")
         <<",\n  \"failure_threshold_m\": "<<threshold_
         <<",\n  \"frames_evaluated\": "<<frame_error_.size()
         <<",\n  \"duration_s\": "<<m.length<<",\n  \"evaluated_s\": "<<elapsed
         <<",\n  \"first_failure_s\": "<<(first_failure_<0?"null":std::to_string(first_failure_))
         <<",\n  \"mean_position_error_m\": "<<mean(frame_error_)
         <<",\n  \"p95_body_position_error_m\": "<<percentile(body_error_,.95)
         <<",\n  \"max_mean_position_error_m\": "<<percentile(frame_error_,1)
         <<",\n  \"mean_rotation_error_rad\": "<<mean(rotation_error_)
         <<",\n  \"reference_kinematics\": {\"peak_body_speed_m_s\": "<<peak_speed
         <<", \"peak_root_speed_m_s\": "<<peak_root<<", \"peak_body_angular_speed_rad_s\": "<<peak_ang
         <<", \"peak_joint_speed_rad_s\": "<<peak_joint<<", \"peak_body_acceleration_m_s2\": "<<peak_accel<<"}"
         <<",\n  \"limbs\": {";
        const char* names[]={"left_arm","right_arm","left_leg","right_leg","trunk"};
        for(int i=0;i<5;++i)f<<(i?",":"")<<"\n    "<<quote(names[i])<<": {\"mean_position_error_m\": "<<mean(limb_error_[i])<<", \"p95_position_error_m\": "<<percentile(limb_error_[i],.95)<<"}";
        f<<"\n  }\n}\n";f.close();if(!f)throw std::runtime_error("Failed writing evaluation report");
        std::filesystem::rename(tmp,dest);
    }
};
} // namespace ovpp
