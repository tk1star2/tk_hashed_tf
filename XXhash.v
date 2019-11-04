`timescale 1ns / 1ps

module XXhash(
    input clk,
    input rst,
    input [15:0] idx_in_col,
    input [15:0] idx_in_row,
    
    output reg [4:0] centroid
    );
    parameter PRIME32_1=32'h9E3779B1;
    parameter PRIME32_2=32'h85EBCA77;
    parameter PRIME32_3=32'hC2B2AE3D;
    parameter PRIME32_4=32'h27D4EB2F;
    parameter PRIME32_5=32'h165667B1;
    
    parameter seed=32'h00000000;
    parameter inputLength=32'h00010000;
    
    parameter acc1 = seed + PRIME32_1 + PRIME32_2;
    parameter acc2 = seed + PRIME32_2;
    parameter acc3 = seed;
    parameter acc4 = seed - PRIME32_1;
    
    // (1,100) -> 1, 2, 3, 100, 101, 102
    //reg [31:0] acc1 = seed + PRIME32_1 + PRIME32_2;
    //reg [31:0] acc2 = seed + PRIME32_2;
    //reg [31:0] acc3 = seed;
    //reg [31:0] acc4 = seed - PRIME32_1;
    
    reg [31:0] lane1;
    reg [31:0] lane2;
    reg [31:0] lane3;
    reg [31:0] lane4;
    
    reg [31:0] acc1_w;
    reg [31:0] acc2_w;
    reg [31:0] acc3_w;
    reg [31:0] acc4_w;
    
    reg [31:0] acc_temp;
    
    reg [31:0] acc;
    reg [31:0] centroid_temp;
    
    
    //step1
    always @(posedge clk) begin
        if(rst) begin
            lane1 <= 32'd0;
            lane2 <= 32'd0;
            lane3 <= 32'd0;
            lane4 <= 32'd0;
        end else begin
            lane1 <= {16'h0000, idx_in_col };
            lane2 <= {16'h0000, idx_in_col+1};
            lane3 <= {16'h0000, idx_in_row};
            lane4 <= {16'h0000, idx_in_row+1};
        end
    end
    
    always @(*) begin
        acc1_w = acc1 + (lane1 * PRIME32_2);
        acc2_w = acc2 + (lane2 * PRIME32_2);
        acc3_w = acc3 + (lane3 * PRIME32_2);
        acc4_w = acc4 + (lane4 * PRIME32_2);
     end
    always @(*) begin
         acc_temp = (acc1_w << 1) + (acc2_w << 7) + (acc3 << 12) + (acc4 << 18) + inputLength;
    end
     
     //step2
     always @(posedge clk) begin
        if(rst) begin
            acc <= 32'd0;
        end else begin
            acc <= acc_temp;
        end 
     end
     
     always @(*) begin
        centroid_temp = ((((acc ^ (acc >>15)) * PRIME32_2) ^ (acc >> 13)) * PRIME32_3) ^ (acc >> 16);
     end
     
     always @(*) begin
        centroid = centroid_temp[4:0];
     end
endmodule
