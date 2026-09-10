#include "od_p3pf/od_p3pf.hpp"
#include "od_p3pf/od_p3pf_gb.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
using od_p3pf::Mat3;
using od_p3pf::Solution;
using od_p3pf::Vec2;
using od_p3pf::Vec3;
constexpr double kPi = 3.14159265358979323846;

Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 transpose_mul(const Mat3& a, const Vec3& x) {
  return {a(0,0)*x.x+a(1,0)*x.y+a(2,0)*x.z,
          a(0,1)*x.x+a(1,1)*x.y+a(2,1)*x.z,
          a(0,2)*x.x+a(1,2)*x.y+a(2,2)*x.z};
}
double norm(const Vec3& a) { return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
double rotation_error(const Mat3& a, const Mat3& b) {
  double trace = 0.0;
  for (int i=0;i<3;++i) for(int j=0;j<3;++j) trace += a(i,j)*b(i,j);
  return 180.0/kPi*std::acos(std::clamp((trace-1.0)/2.0,-1.0,1.0));
}
Mat3 random_rotation(std::mt19937_64& rng) {
  std::normal_distribution<double> n(0.0,1.0);
  double w=n(rng),x=n(rng),y=n(rng),z=n(rng);
  const double s=1.0/std::sqrt(w*w+x*x+y*y+z*z); w*=s;x*=s;y*=s;z*=s;
  Mat3 r;
  r(0,0)=1-2*(y*y+z*z); r(0,1)=2*(x*y-z*w); r(0,2)=2*(x*z+y*w);
  r(1,0)=2*(x*y+z*w); r(1,1)=1-2*(x*x+z*z); r(1,2)=2*(y*z-x*w);
  r(2,0)=2*(x*z-y*w); r(2,1)=2*(y*z+x*w); r(2,2)=1-2*(x*x+y*y);
  return r;
}
struct Scene { std::array<Vec3,3> world,camera; std::array<Vec2,3> pixels; Mat3 R; Vec3 t; double f; };
Scene make_scene(std::mt19937_64& rng,double image_noise,double depth_noise) {
  std::uniform_real_distribution<double> fd(500,1500),xd(-2.2,2.2),yd(-1.6,1.6),zd(4,10),td(-2,2);
  std::normal_distribution<double> n(0,1);
  Scene s; s.f=fd(rng); s.R=random_rotation(rng); s.t={td(rng),td(rng),td(rng)};
  for(int i=0;i<3;++i){ s.camera[i]={xd(rng),yd(rng),zd(rng)}; s.world[i]=transpose_mul(s.R,s.camera[i]-s.t);
    s.pixels[i]={s.f*s.camera[i].x/s.camera[i].z+image_noise*n(rng),s.f*s.camera[i].y/s.camera[i].z+image_noise*n(rng)}; }
  s.camera[0].z*=1.0+depth_noise*n(rng); return s;
}
double median(std::vector<double> v){ if(v.empty())return std::numeric_limits<double>::quiet_NaN();
  const size_t m=v.size()/2; std::nth_element(v.begin(),v.begin()+m,v.end()); return v[m]; }
struct Summary { int empty=0; std::vector<double> rot,trans,focal,count; };
void add_best(const Scene& s,const std::vector<Solution>& sols,Summary* out){ out->count.push_back((double)sols.size());
  if(sols.empty()){++out->empty;return;} const Solution* best=nullptr; double bs=1e300;
  for(const auto& q:sols){ double r=rotation_error(q.R,s.R),t=norm(q.t-s.t),f=std::abs(q.focal-s.f)/s.f;
    double score=r+t+10*f; if(score<bs){bs=score;best=&q;} }
  out->rot.push_back(rotation_error(best->R,s.R)); out->trans.push_back(norm(best->t-s.t)); out->focal.push_back(std::abs(best->focal-s.f)/s.f); }
void print_summary(const char* label,const Summary& s,int n){ std::cout<<label<<",trials="<<n<<",empty="<<s.empty
  <<",median_count="<<median(s.count)<<",median_rot_deg="<<median(s.rot)<<",median_trans="<<median(s.trans)
  <<",median_focal_rel="<<median(s.focal)<<"\n"; }
}

int main(int argc,char** argv){ const bool quick=argc>1&&std::string(argv[1])=="--quick"; const int trials=quick?200:10000;
  {
    const std::array<Vec2,3> rays{{{0.016217,-0.524291},{2.154191,-3.615259},{-0.554594,-1.396616}}};
    const std::array<double,3> depth{{5.326722,1.795670,4.218296}};
    std::array<Vec3,3> world{}; std::array<Vec2,3> pixels{};
    for(int i=0;i<3;++i){world[i]={depth[i]*rays[i].x,depth[i]*rays[i].y,depth[i]};pixels[i]={1000*rays[i].x,1000*rays[i].y};}
    std::cout<<"six_solution_witness,direct="<<od_p3pf::solve(world,pixels,depth[0]).size()
             <<",gb="<<od_p3pf::solve_gb(world,pixels,depth[0]).size()<<"\n";
  }
  std::mt19937_64 rng(20260908); std::vector<Scene> exact,noisy; exact.reserve(trials);noisy.reserve(trials);
  for(int i=0;i<trials;++i){exact.push_back(make_scene(rng,0,0));noisy.push_back(make_scene(rng,1.0,0.005));}
  for(const auto& corpus: {std::cref(exact),std::cref(noisy)}){ const bool is_exact=&corpus.get()==&exact; Summary d,g;
    for(const auto& s:corpus.get()){add_best(s,od_p3pf::solve(s.world,s.pixels,s.camera[0].z),&d);add_best(s,od_p3pf::solve_gb(s.world,s.pixels,s.camera[0].z),&g);}
    print_summary(is_exact?"direct_exact":"direct_noisy",d,trials); print_summary(is_exact?"gb_exact":"gb_noisy",g,trials); }
  const int reps=quick?1000:100000; std::vector<Scene> bench;bench.reserve(256);for(int i=0;i<256;++i)bench.push_back(make_scene(rng,1.0,0.005));
  volatile std::size_t sink=0; auto t0=std::chrono::steady_clock::now();for(int i=0;i<reps;++i)sink+=od_p3pf::solve(bench[i%bench.size()].world,bench[i%bench.size()].pixels,bench[i%bench.size()].camera[0].z).size();
  auto t1=std::chrono::steady_clock::now();for(int i=0;i<reps;++i)sink+=od_p3pf::solve_gb(bench[i%bench.size()].world,bench[i%bench.size()].pixels,bench[i%bench.size()].camera[0].z).size();auto t2=std::chrono::steady_clock::now();
  const double du=std::chrono::duration<double,std::micro>(t1-t0).count()/reps,gu=std::chrono::duration<double,std::micro>(t2-t1).count()/reps;
  std::cout<<std::fixed<<std::setprecision(6)<<"runtime,repetitions="<<reps<<",direct_us="<<du<<",gb_us="<<gu<<",ratio="<<gu/du<<",sink="<<sink<<"\n";
  return 0; }
