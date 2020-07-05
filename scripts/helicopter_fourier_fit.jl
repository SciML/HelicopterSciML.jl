using Pkg
cd(@__DIR__)
Pkg.activate("."); Pkg.instantiate()

using Plots; gr()
using LaTeXStrings
using DataFrames
using CSV
using OrdinaryDiffEq
using LinearAlgebra
using DataInterpolations
using BenchmarkTools
using DiffEqFlux
using Optim
using Flux
using Zygote
using ApproxFun

# Model parameters
rc = 1.5e-2 # m
h = 0.5e-2 # m
m = 0.5 # kg
Jbc = 15e-3 # kg.m2
Js = 5e-3 # kg.m2
g = 9.81 # m/s2
K_th_th = 55e-3 # Nm/V
K_th_psi = 5e-3 # Nm/V
K_psi_th = 0.15 # Nm/V
K_psi_psi = 0.2 # Nm/V
D_th = 1e-2 # Nm/(rad/s)
D_psi = 8e-2 # Nm/(rad/s)

# Graphics
figpath = "../figs/"
LW1 = 2.5
LW2 = 1.5
LS1 = :solid
LS2 = :dot
LS3 = :dash
LC1 = :blue
LC2 = :red
LC3 = :green
LC4 = :black
LA1 = 1.0
LA2 = 0.7
LA3 = 0.3

# Parse data from table
df=CSV.read("../data/Lab-Helicopter_Experimental-data.csv")
ENV["COLUMNS"]=100 # change the number of IJulia columns from default 80 to 100
df[1:5,:]
u_th = df[:,1]
u_psi = df[:,2]
theta = df[:,3]
psi = df[:,4]
dt = df[1,5]
tm = 0:dt:(length(psi)-1)*dt
tspan = (0.0, (length(psi)-1)*dt)
Nd = length(tm);
timeseriesdata = [theta';psi']

A = fill(FourierBasis(2), 6)
nn1 = TensorLayer(A, 2)
N1 = size(nn1.p)[1]
α = zeros(N1+6)

function helicopter_uode(dx,x,Ps,P0,u,p,t)
# P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi);
    th = x[1]
    psi = x[2]
    w_th = x[3]
    w_psi = x[4]

    U = [u.th(t),u.psi(t)]
    #
    M11 = Ps[4]*P0.Jbc + Ps[3]*P0.m*((Ps[1]*P0.rc)^2+(Ps[2]*P0.h)^2)
    M22 = Ps[4]*P0.Jbc*cos(th)^2 + Ps[3]*P0.m*(Ps[1]*P0.rc*cos(th)-Ps[2]*P0.h*sin(th))^2 + Ps[5]*P0.Js
    #
    dLdth = -w_psi^2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th)-(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2)) - (Ps[3]*P0.m)*(Ps[6]*P0.g)*((Ps[1]*P0.rc)*cos(th) - (Ps[2]*P0.h)*sin(th))
    #
    dp_psi_dth = -2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th) + (Ps[3]*P0.m)*(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2))*w_psi
    #
    F_th = Ps[7]*P0.K_th_th*U[1] - Ps[8]*P0.K_th_psi*U[2] - P0.D_th*(w_th) + nn1([p[1]*U[1], p[2]*U[2], p[3]*th, p[4]*psi, p[5]*w_th, p[6]*w_psi], p[7:end])[1]
    F_psi = Ps[9]*P0.K_psi_th*U[1] - Ps[10]*P0.K_psi_psi*U[2] - P0.D_psi*(w_psi) + nn1([p[1]*U[1], p[2]*U[2], p[3]*th, p[4]*psi, p[5]*w_th, p[6]*w_psi], p[7:end])[2]

    #
    dx[1] = w_th
    dx[2] = w_psi
    dx[3] = (dLdth + F_th)/M11
    dx[4] = (-dp_psi_dth*w_th + F_psi)/M22
end

P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi)
Ps = fill(1.0,length(P0))
#
th0 = theta[1]
psi0 = psi[1]
w_th0 = (theta[2]-theta[1])/dt
w_psi0 = (psi[2]-psi[1])/dt
#
x0 = [th0,psi0,w_th0,w_psi0]
U_th = ZeroSpline(u_th,tm)
U_psi = ZeroSpline(u_psi,tm)
u_t = (th=U_th, psi=U_psi)
helicopter! = (dx,x,p,t) -> helicopter_uode(dx,x,Ps,P0,u_t,p,t)

prob = ODEProblem(helicopter!,x0,tspan,p=nothing)
function uode_cost(p)
    #
    sol = solve(prob, p=p, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)
    #
    return norm(sol[1:2,1:end]-timeseriesdata[:,1:size(sol,2)])^2/Nd
    #
end

rad2deg(t,u) = (t,u*180/pi)

callback = function (p, l)
    @show p, l
    prob = ODEProblem(helicopter!,x0,tspan,p)
    sol = solve(prob, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)
    pl = plot(sol,vars=(rad2deg,0,2),lw=LW1,lc=LC1)
    plot!(pl,tm,psi*180/pi,lw=LW1,ls=LS2,lc=LC2)
    plot!(pl,xlim = tspan)
    plot!(pl)
    display(pl)
    return false
end
res = DiffEqFlux.sciml_train(uode_cost, α, ADAM(0.01), cb = callback, maxiters = 150)
p = res.minimizer

prob = ODEProblem(helicopter!,x0,tspan,p)
sol_final = solve(prob, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)

plot(sol_final,vars=(rad2deg,0,1),lw=LW1,lc=LC1,label=L"$\theta$")
plot!(tm,theta*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\theta^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\theta$ [$^\circ$]")
plot!(title="Pitch angle: model+TensorLayer(u) (blue) vs. data (red)",box=true)
figname="Helicopter_pitch-angle_model_fit_tensor_layer_u.svg"
savefig(figpath*figname)

plot(sol_final,vars=(rad2deg,0,2),lw=LW1,lc=LC1,label=L"$\theta$")
plot!(tm,psi*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\theta^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\theta$ [$^\circ$]")
plot!(title="Yaw angle: model+TensorLayer(u) (blue) vs. data (red)",box=true)
figname="Helicopter_yaw-angle_model_fit_tensor_layer_u.svg"
savefig(figpath*figname)
