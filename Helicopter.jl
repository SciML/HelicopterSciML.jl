using Pkg
cd(@__DIR__)
Pkg.activate("."); Pkg.instantiate()

using Plots; gr()
using LaTeXStrings
using DataFrames
using CSV
using OMJulia
using OrdinaryDiffEq
using LinearAlgebra
using ControlSystems
using DataInterpolations
using BlackBoxOptim
using BenchmarkTools

# Plot path
figpath = "figs/";
# Linewidths and styles
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
;

# Function for helicopter model
function helicopter_ode(dx,x,Ps,t,P0,u)
# P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi);
    th = x[1]
    psi = x[2]
    w_th = x[3]
    w_psi = x[4]
    #
    M11 = Ps[4]*P0.Jbc + Ps[3]*P0.m*((Ps[1]*P0.rc)^2+(Ps[2]*P0.h)^2)
    M22 = Ps[4]*P0.Jbc*cos(th)^2 + Ps[3]*P0.m*(Ps[1]*P0.rc*cos(th)-Ps[2]*P0.h*sin(th))^2 + Ps[5]*P0.Js
    #
    dLdth = -w_psi^2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th)-(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2)) - (Ps[3]*P0.m)*(Ps[6]*P0.g)*((Ps[1]*P0.rc)*cos(th) - (Ps[2]*P0.h)*sin(th))
    #
    dp_psi_dth = -2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th) + (Ps[3]*P0.m)*(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2))*w_psi
    #
    F_th = Ps[7]*P0.K_th_th*u.th(t) - Ps[8]*P0.K_th_psi*u.psi(t) - P0.D_th*w_th
    F_psi = Ps[9]*P0.K_psi_th*u.th(t) - Ps[10]*P0.K_psi_psi*u.psi(t) - P0.D_psi*w_psi
    #
    dx[1] = w_th
    dx[2] = w_psi
    dx[3] = (dLdth + F_th)/M11
    dx[4] = (-dp_psi_dth*w_th + F_psi)/M22
end
#
# Converting from radians to degrees in plots
#
rad2deg(t,u) = (t,u*180/pi)
;

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
#
P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi)
Ps = fill(1.0,length(P0))
;

# Set-up for simulating helicopter
#
th0 = pi/2
psi0 = 0.0
w_th0 = -0.1
w_psi0 = 0.1
#
x0 = [th0,psi0,w_th0,w_psi0]
u_c = (th=t->0.0, psi=t->0.0)
#
helicopter! = (dx,x,Ps,t) -> helicopter_ode(dx,x,Ps,t,P0,u_c)
#
tspan = (0.,20)
prob = ODEProblem(helicopter!,x0,tspan,Ps)
#
sol = solve(prob,AutoTsit5(Rosenbrock23()),reltol=1e-6)
#
plot(sol,vars=(rad2deg,0,1),lw=LW1,lc=LC1,label=L"$\theta$ [deg]")
plot!(sol,vars=(rad2deg,0,2),lw=LW1,lc=LC2,label=L"$\psi$ [deg]")
plot!(xlim = tspan,xlabel=L"time $t$ [s]")
figname="Helicopter_theoretical_simulation.svg"
savefig(figpath*figname)

df=CSV.read("data/Lab-Helicopter_Experimental-data.csv")
ENV["COLUMNS"]=100 # change the number of IJulia columns from default 80 to 100
df[1:5,:]

u_th = df[:,1]
u_psi = df[:,2]
theta = df[:,3]
psi = df[:,4]
dt = df[1,5]
tm = 0:dt:(length(psi)-1)*dt
;

data = [theta';psi']
Nd = length(tm);

plot(tm,theta*180/pi,lw=LW2,lc=LC1,label=L"\theta",legend=:topleft)
plot!(twinx(),tm,psi*180/pi,lw=LW2,lc=LC2,label=L"\psi")
plot!(title="Experimental helicopter outputs: measured angles", ylabel=L"angles [$^\circ$]", xlabel=L"time $t$ [s]", box=true)
figname="Helicopter_experimental_angles.svg"
savefig(figpath*figname)

plot(tm,u_th,lw=LW1,lc=LC1,label=L"u_\theta",legend=:topleft)
plot!(twinx(),tm,u_psi,lw=LW1,lc=LC2,label=L"u_\psi",legend=:bottomright)
plot!(title="Experimental helicopter inputs: motor voltages", ylabel=L"$u$ [V]", xlabel=L"time $t$ [s]",box=true)
figname="Helicopter_experimental_voltages.svg"
savefig(figpath*figname)

U_th = ConstantInterpolation(u_th,tm)
U_psi = ConstantInterpolation(u_psi,tm)
Theta = ConstantInterpolation(theta,tm)
Psi = ConstantInterpolation(psi,tm);

# Set-up for simulating helicopter
#
th0 = Theta(0.0)
psi0 = Psi(0.0)
w_th0 = 0.0
w_psi0 = 0.0
#
x0 = [th0,psi0,w_th0,w_psi0]
u_t = (th=U_th, psi=U_psi)
#
helicopter! = (dx,x,Ps,t) -> helicopter_ode(dx,x,Ps,t,P0,u_t)
#
tspan = (tm[1],tm[end])
#
prob = ODEProblem(helicopter!,x0,tspan,Ps)
#
sol = solve(prob,AutoTsit5(Rosenbrock23()),reltol=1e-6)
#
plot(sol,vars=(rad2deg,0,1),lw=LW1,lc=LC1,label=L"$\theta$ [deg]")
plot!(sol,vars=(rad2deg,0,2),lw=LW1,lc=LC2,label=L"$\psi$ [deg]")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel="angles")
plot!(title="Angles: experimental inputs, nominal parameters",box=true)
figname="Helicopter_simulated_angles_experimental_inputs_nominal.svg"
savefig(figpath*figname)

function cost(p,Ps,pidx,x0,xidx,data,Nd,tspan)
    #
    xx0 = copy(x0)
    xx0[xidx] = p[xidx].*x0[xidx]
    Ps[pidx] .= p[length(xidx)+1:end]
    #
    prob = ODEProblem(helicopter!,xx0,tspan,Ps)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=dt)
    #
    return norm(sol[1:2,:]-data,2)^2/Nd
    #
end
#
;

#
# (:rc, :h, :m, :Jbc, :Js, :g, :K_th_th, :K_th_psi, :K_psi_th, :K_psi_psi, :D_th, :D_psi)
Ps_keys = [:rc, :h, :m, :Jbc, :Js, :K_th_th,:K_th_psi,:K_psi_th,:K_psi_psi,:D_th, :D_psi]
pidx = [findall(Ps_keys[i] .== keys(P0))[1] for i in 1:length(Ps_keys)]
#
xidx = collect(3:4)
#
p = fill(1.0,length(xidx)+length(pidx))
#
p_lo = 0.2*p
p_hi = 2*p
p_lo_hi = collect(zip(p_lo,p_hi))
#
# Initial value of loss function
Ps = fill(1.0,length(P0))
x0 = [Theta(0),Psi(0),1.0,1.0]
#
loss = (p) -> cost(p,Ps,pidx,x0,xidx,data,Nd,tspan)
;

loss(p)

@time res = bboptimize(loss; SearchRange = p_lo_hi, NumDimensions = length(p));

best_fitness(res)

p_best = best_candidate(res)

loss(p_best)

x0_best = copy(x0)
x0_best[xidx] = p_best[xidx].*x0[xidx]
Ps_best=fill(1.0,length(P0))
Ps_best[pidx] .= p_best[length(xidx)+1:end]
#
prob_best = ODEProblem(helicopter!,x0_best,tspan,Ps_best)
#
sol_best = solve(prob_best,AutoTsit5(Rosenbrock23()),reltol=1e-6)
#
plot(sol_best,vars=(rad2deg,0,1),lw=LW1,lc=LC1,label=L"$\theta$ [deg]")
plot!(tm,theta*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\theta^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\theta$ [$^\circ$]")
plot!(title="Pitch angle: model (blue, solid) vs. data (red, dotted)",box=true)
figname="Helicopter_pitch-angle_model_fit.svg"
savefig(figpath*figname)

plot(sol_best,vars=(rad2deg,0,2),lw=LW1,lc=LC1,label=L"$\psi$ [deg]")
plot!(tm,psi*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\psi^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\psi$ [$^\circ$]")
plot!(title="Yaw angle: model (blue, solid) vs. data (red, dotted)",box=true)
figname="Helicopter_yaw-angle_model_fit.svg"
savefig(figpath*figname)

# Estimated initial states -- states 3 and 4 have been estimated
x0_best[xidx] = p_best[xidx].*x0[xidx]
show(x0_best)

# Estimated model parameters -- all parameters except gravity g have been assumed uncertain
Ps_best=fill(1.0,length(P0))
Ps_best[pidx] .= p_best[length(xidx)+1:end]
collect(zip(keys(P0),Ps_best.*values(P0)))

# Nominal parameters, taken from laboratory task description (and rounded somewhat)
P0

open("optimization_results/globaloptimizationresults.jl", "w") do f
    write(f,"P0 = $P0 \n")
    write(f,"Ps_best = $Ps_best \n")
    write(f,"x0_best = $x0_best \n")
    write(f,"tspan = $tspan \n")
end
