/* Generated by Yosys 0.39 (git sha1 00338082b, g++ 13.2.1 -march=x86-64 -mtune=generic -O2 -fno-plt -fexceptions -fstack-clash-protection -fcf-protection -ffile-prefix-map=/build/yosys/src=/usr/src/debug/yosys -fPIC -Os) */

(* top =  1  *)
(* src = "./add8s_83C.v:18.1-90.10" *)
module add8s_83C(A, B, O);
  wire _000_;
  wire _001_;
  wire _002_;
  wire _003_;
  wire _004_;
  wire _005_;
  wire _006_;
  wire _007_;
  wire _008_;
  wire _009_;
  wire _010_;
  wire _011_;
  wire _012_;
  wire _013_;
  wire _014_;
  wire _015_;
  wire _016_;
  wire _017_;
  wire _018_;
  wire _019_;
  wire _020_;
  wire _021_;
  wire _022_;
  wire _023_;
  wire _024_;
  wire _025_;
  wire _026_;
  wire _027_;
  wire _028_;
  wire _029_;
  wire _030_;
  wire _031_;
  wire _032_;
  wire _033_;
  wire _034_;
  wire _035_;
  wire _036_;
  wire _037_;
  wire _038_;
  wire _039_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _040_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _041_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _042_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _043_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _044_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _045_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _046_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  wire _047_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _048_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _049_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _050_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _051_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _052_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _053_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _054_;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  wire _055_;
  wire _056_;
  wire _057_;
  wire _058_;
  wire _059_;
  wire _060_;
  wire _061_;
  wire _062_;
  wire _063_;
  wire _064_;
  wire _065_;
  wire _066_;
  wire _067_;
  wire _068_;
  wire _069_;
  wire _070_;
  wire _071_;
  wire _072_;
  wire _073_;
  wire _074_;
  wire _075_;
  wire _076_;
  wire _077_;
  wire _078_;
  wire _079_;
  wire _080_;
  wire _081_;
  wire _082_;
  wire _083_;
  wire _084_;
  wire _085_;
  wire _086_;
  wire _087_;
  wire _088_;
  wire _089_;
  wire _090_;
  wire _091_;
  wire _092_;
  wire _093_;
  wire _094_;
  wire _095_;
  wire _096_;
  wire _097_;
  wire _098_;
  wire _099_;
  wire _100_;
  wire _101_;
  wire _102_;
  wire _103_;
  wire _104_;
  wire _105_;
  wire _106_;
  wire _107_;
  wire _108_;
  wire _109_;
  wire _110_;
  wire _111_;
  wire _112_;
  wire _113_;
  wire _114_;
  wire _115_;
  wire _116_;
  wire _117_;
  wire _118_;
  wire _119_;
  wire _120_;
  wire _121_;
  wire _122_;
  wire _123_;
  wire _124_;
  wire _125_;
  wire _126_;
  wire _127_;
  wire _128_;
  wire _129_;
  wire _130_;
  wire _131_;
  wire _132_;
  wire _133_;
  wire _134_;
  wire _135_;
  wire _136_;
  wire _137_;
  wire _138_;
  wire _139_;
  (* src = "./add8s_83C.v:28.13-28.19" *)
  wire _140_;
  (* src = "./add8s_83C.v:30.13-30.19" *)
  wire _141_;
  (* src = "./add8s_83C.v:30.20-30.26" *)
  wire _142_;
  (* src = "./add8s_83C.v:30.27-30.33" *)
  wire _143_;
  (* src = "./add8s_83C.v:30.34-30.40" *)
  wire _144_;
  (* src = "./add8s_83C.v:30.41-30.47" *)
  wire _145_;
  (* src = "./add8s_83C.v:30.48-30.54" *)
  wire _146_;
  (* src = "./add8s_83C.v:30.55-30.61" *)
  wire _147_;
  (* src = "./add8s_83C.v:24.13-24.14" *)
  input [7:0] A;
  wire [7:0] A;
  (* src = "./add8s_83C.v:25.13-25.14" *)
  input [7:0] B;
  wire [7:0] B;
  (* src = "./add8s_83C.v:26.14-26.15" *)
  output [7:0] O;
  wire [7:0] O;
  (* src = "./add8s_83C.v:28.13-28.19" *)
  wire sig_17;
  (* src = "./add8s_83C.v:30.13-30.19" *)
  wire sig_58;
  (* src = "./add8s_83C.v:30.20-30.26" *)
  wire sig_59;
  (* src = "./add8s_83C.v:30.27-30.33" *)
  wire sig_60;
  (* src = "./add8s_83C.v:30.34-30.40" *)
  wire sig_61;
  (* src = "./add8s_83C.v:30.41-30.47" *)
  wire sig_62;
  (* src = "./add8s_83C.v:30.48-30.54" *)
  wire sig_63;
  (* src = "./add8s_83C.v:30.55-30.61" *)
  wire sig_64;
  assign _071_ = ~_048_;
  assign _072_ = ~_040_;
  assign _073_ = ~_049_;
  assign _074_ = ~_041_;
  assign _075_ = ~_050_;
  assign _076_ = ~_042_;
  assign _077_ = ~_051_;
  assign _078_ = ~_043_;
  assign _079_ = ~_052_;
  assign _080_ = ~_044_;
  assign _081_ = ~_053_;
  assign _082_ = ~_045_;
  assign _083_ = ~_054_;
  assign _084_ = ~_046_;
  assign _085_ = ~_055_;
  assign _086_ = ~_047_;
  assign _087_ = ~(_048_ & _040_);
  assign _088_ = ~_087_;
  assign _089_ = ~(_071_ & _072_);
  assign _090_ = ~(_087_ & _089_);
  assign _140_ = ~_090_;
  assign _091_ = ~(_073_ & _074_);
  assign _092_ = ~(_049_ & _041_);
  assign _093_ = ~(_091_ & _092_);
  assign _094_ = ~_093_;
  assign _095_ = ~(_088_ & _094_);
  assign _096_ = ~(_087_ & _093_);
  assign _097_ = ~(_095_ & _096_);
  assign _141_ = ~_097_;
  assign _098_ = ~(_092_ & _095_);
  assign _099_ = ~_098_;
  assign _100_ = ~(_075_ & _076_);
  assign _101_ = ~(_050_ & _042_);
  assign _102_ = ~(_100_ & _101_);
  assign _103_ = ~_102_;
  assign _104_ = ~(_098_ & _103_);
  assign _105_ = ~(_099_ & _102_);
  assign _106_ = ~(_104_ & _105_);
  assign _142_ = ~_106_;
  assign _107_ = ~(_101_ & _104_);
  assign _108_ = ~_107_;
  assign _109_ = ~(_077_ & _078_);
  assign _110_ = ~(_051_ & _043_);
  assign _111_ = ~(_109_ & _110_);
  assign _112_ = ~_111_;
  assign _113_ = ~(_107_ & _112_);
  assign _114_ = ~(_108_ & _111_);
  assign _115_ = ~(_113_ & _114_);
  assign _143_ = ~_115_;
  assign _116_ = ~(_110_ & _113_);
  assign _117_ = ~_116_;
  assign _118_ = ~(_079_ & _080_);
  assign _119_ = ~(_052_ & _044_);
  assign _120_ = ~(_118_ & _119_);
  assign _121_ = ~_120_;
  assign _122_ = ~(_117_ & _121_);
  assign _123_ = ~(_116_ & _120_);
  assign _144_ = ~(_122_ & _123_);
  assign _124_ = ~(_081_ & _082_);
  assign _125_ = ~(_053_ & _045_);
  assign _126_ = ~(_124_ & _125_);
  assign _127_ = ~_126_;
  assign _128_ = ~(_116_ & _118_);
  assign _129_ = ~(_117_ & _119_);
  assign _130_ = ~(_118_ & _129_);
  assign _131_ = ~(_119_ & _128_);
  assign _132_ = ~(_127_ & _131_);
  assign _133_ = ~(_126_ & _130_);
  assign _134_ = ~(_132_ & _133_);
  assign _145_ = ~_134_;
  assign _135_ = ~(_125_ & _132_);
  assign _136_ = ~_135_;
  assign _137_ = ~(_083_ & _084_);
  assign _138_ = ~(_054_ & _046_);
  assign _139_ = ~(_137_ & _138_);
  assign _056_ = ~_139_;
  assign _057_ = ~(_135_ & _139_);
  assign _058_ = ~(_136_ & _056_);
  assign _146_ = ~(_057_ & _058_);
  assign _059_ = ~(_135_ & _137_);
  assign _060_ = ~(_136_ & _138_);
  assign _061_ = ~(_137_ & _060_);
  assign _062_ = ~(_138_ & _059_);
  assign _063_ = ~(_085_ & _086_);
  assign _064_ = ~(_055_ & _047_);
  assign _065_ = ~(_085_ & _047_);
  assign _066_ = ~(_055_ & _086_);
  assign _067_ = ~(_065_ & _066_);
  assign _068_ = ~(_063_ & _064_);
  assign _069_ = ~(_061_ & _067_);
  assign _070_ = ~(_062_ & _068_);
  assign _147_ = ~(_069_ & _070_);
  assign O = { sig_64, sig_63, sig_62, sig_61, sig_60, sig_59, sig_58, sig_17 };
  assign _048_ = B[0];
  assign _040_ = A[0];
  assign sig_17 = _140_;
  assign _049_ = B[1];
  assign _041_ = A[1];
  assign sig_58 = _141_;
  assign _050_ = B[2];
  assign _042_ = A[2];
  assign sig_59 = _142_;
  assign _051_ = B[3];
  assign _043_ = A[3];
  assign sig_60 = _143_;
  assign _052_ = B[4];
  assign _044_ = A[4];
  assign sig_61 = _144_;
  assign _053_ = B[5];
  assign _045_ = A[5];
  assign sig_62 = _145_;
  assign _054_ = B[6];
  assign _046_ = A[6];
  assign sig_63 = _146_;
  assign _055_ = B[7];
  assign _047_ = A[7];
  assign sig_64 = _147_;
endmodule
