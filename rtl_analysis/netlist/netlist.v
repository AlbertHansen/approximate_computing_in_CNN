/* Generated by Yosys 0.9 (git sha1 1979e0b) */

(* top =  1  *)
(* src = "./add8se_8VQ/add8se_8VQ.v:18" *)
module add8se_8VQ(A, B, O);
  wire _000_;
  wire _001_;
  wire _002_;
  wire _003_;
  wire _004_;
  wire _005_;
  wire _006_;
  wire _007_;
  wire _008_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:24" *)
  wire _009_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:24" *)
  wire _010_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:24" *)
  wire _011_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:24" *)
  wire _012_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:25" *)
  wire _013_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:25" *)
  wire _014_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:25" *)
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
  wire _040_;
  wire _041_;
  wire _042_;
  wire _043_;
  wire _044_;
  wire _045_;
  wire _046_;
  wire _047_;
  wire _048_;
  wire _049_;
  wire _050_;
  wire _051_;
  wire _052_;
  wire _053_;
  wire _054_;
  wire _055_;
  wire _056_;
  wire _057_;
  wire _058_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire _059_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire _060_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire _061_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire _062_;
  (* src = "./add8se_8VQ/add8se_8VQ.v:24" *)
  input [7:0] A;
  (* src = "./add8se_8VQ/add8se_8VQ.v:25" *)
  input [7:0] B;
  (* src = "./add8se_8VQ/add8se_8VQ.v:26" *)
  output [8:0] O;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire sig_41;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire sig_46;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire sig_51;
  (* src = "./add8se_8VQ/add8se_8VQ.v:28" *)
  wire sig_54;
  assign _023_ = _016_ | _017_;
  assign _024_ = ~_023_;
  assign _025_ = _013_ | _010_;
  assign _026_ = ~_025_;
  assign _027_ = _024_ | _026_;
  assign _028_ = ~_027_;
  assign _029_ = _018_ | _027_;
  assign _030_ = ~_029_;
  assign _031_ = _009_ | _028_;
  assign _032_ = ~_031_;
  assign _033_ = _030_ | _032_;
  assign _059_ = ~_033_;
  assign _034_ = _024_ | _030_;
  assign _035_ = ~_034_;
  assign _036_ = _014_ | _011_;
  assign _037_ = ~_036_;
  assign _038_ = _019_ | _020_;
  assign _039_ = ~_038_;
  assign _040_ = _037_ | _039_;
  assign _041_ = ~_040_;
  assign _042_ = _035_ | _040_;
  assign _043_ = ~_042_;
  assign _044_ = _034_ | _041_;
  assign _045_ = ~_044_;
  assign _046_ = _043_ | _045_;
  assign _060_ = ~_046_;
  assign _047_ = _039_ | _043_;
  assign _048_ = ~_047_;
  assign _049_ = _021_ | _022_;
  assign _050_ = ~_049_;
  assign _051_ = _015_ | _012_;
  assign _052_ = ~_051_;
  assign _053_ = _050_ | _052_;
  assign _054_ = ~_053_;
  assign _055_ = _047_ | _053_;
  assign _056_ = ~_055_;
  assign _057_ = _048_ | _054_;
  assign _058_ = ~_057_;
  assign _061_ = _056_ | _058_;
  assign _062_ = _050_ | _056_;
  assign _016_ = ~_013_;
  assign _017_ = ~_010_;
  assign _018_ = ~_009_;
  assign _019_ = ~_014_;
  assign _020_ = ~_011_;
  assign _021_ = ~_015_;
  assign _022_ = ~_012_;
  assign O = { sig_54, sig_51, sig_46, sig_41, B[4:3], A[2], A[3], A[3] };
  assign _013_ = B[5];
  assign _010_ = A[5];
  assign _009_ = A[4];
  assign sig_41 = _059_;
  assign _014_ = B[6];
  assign _011_ = A[6];
  assign sig_46 = _060_;
  assign _015_ = B[7];
  assign _012_ = A[7];
  assign sig_51 = _061_;
  assign sig_54 = _062_;
endmodule
