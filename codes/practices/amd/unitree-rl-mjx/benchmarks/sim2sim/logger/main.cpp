// CSV logger for the sim2sim closed loop: subscribes the bridge's lowstate,
// sportmodestate, and wirelesscontroller topics and stamps every row with
// wall-clock nanoseconds so the tracking analysis can align states with the
// scripted command segments.

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

namespace {

volatile std::sig_atomic_t stop_flag = 0;
void handle_signal(int) { stop_flag = 1; }

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::ofstream low_csv, high_csv, wireless_csv;

void on_lowstate(const void *message) {
  const auto &s = *static_cast<const unitree_go::msg::dds_::LowState_ *>(message);
  low_csv << now_ns() << "," << s.tick();
  for (int i = 0; i < 4; ++i) low_csv << "," << s.imu_state().quaternion()[i];
  for (int i = 0; i < 3; ++i) low_csv << "," << s.imu_state().gyroscope()[i];
  for (int i = 0; i < 12; ++i) low_csv << "," << s.motor_state()[i].q();
  for (int i = 0; i < 12; ++i) low_csv << "," << s.motor_state()[i].dq();
  for (int i = 0; i < 12; ++i) low_csv << "," << s.motor_state()[i].tau_est();
  low_csv << "\n";
}

void on_highstate(const void *message) {
  const auto &s =
      *static_cast<const unitree_go::msg::dds_::SportModeState_ *>(message);
  high_csv << now_ns();
  for (int i = 0; i < 3; ++i) high_csv << "," << s.position()[i];
  for (int i = 0; i < 3; ++i) high_csv << "," << s.velocity()[i];
  high_csv << "\n";
}

void on_wireless(const void *message) {
  const auto &s =
      *static_cast<const unitree_go::msg::dds_::WirelessController_ *>(message);
  wireless_csv << now_ns() << "," << s.lx() << "," << s.ly() << "," << s.rx()
               << "," << s.ry() << "," << s.keys() << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: sim2sim_logger <network_interface> <out_dir>\n";
    return 1;
  }
  std::filesystem::path out_dir(argv[2]);
  std::filesystem::create_directories(out_dir);

  low_csv.open(out_dir / "lowstate.csv");
  low_csv << "t_ns,tick,qw,qx,qy,qz,wx,wy,wz";
  for (int i = 0; i < 12; ++i) low_csv << ",q" << i;
  for (int i = 0; i < 12; ++i) low_csv << ",dq" << i;
  for (int i = 0; i < 12; ++i) low_csv << ",tau" << i;
  low_csv << "\n";
  high_csv.open(out_dir / "highstate.csv");
  high_csv << "t_ns,px,py,pz,vx,vy,vz\n";
  wireless_csv.open(out_dir / "wireless.csv");
  wireless_csv << "t_ns,lx,ly,rx,ry,keys\n";

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

  unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_> low_sub(
      "rt/lowstate");
  low_sub.InitChannel(on_lowstate, 1);
  unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>
      high_sub("rt/sportmodestate");
  high_sub.InitChannel(on_highstate, 1);
  unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>
      wireless_sub("rt/wirelesscontroller");
  wireless_sub.InitChannel(on_wireless, 1);

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  std::cout << "logging to " << out_dir << " (Ctrl-C to stop)\n";
  while (!stop_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  low_csv.close();
  high_csv.close();
  wireless_csv.close();
  return 0;
}
