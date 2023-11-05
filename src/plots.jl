
function plot_trace(
    iters::Vector{FP}, 
    trace::VecOrMat{FP};
    xlabel::String = "Iterations", 
    ylabel::String = "Values", 
    title::String = "Evolution",
    add = false,
    linealpha = nothing,
    linecolor = :auto,
    linestyle = :solid,
    linewidth = :auto,
    marker = :o,
    markersize = 4,
    markershape = :none,
    markeralpha = nothing,
    markercolor = :match,
    markerstrokealpha = nothing,
    markerstrokecolor = :auto,
    markerstrokestyle = :solid,
    markerstrokewidth = 1,
    )

    plt = begin
        if !add
            plot()
            plot!(
                xlabel = xlabel,
                ylabel = ylabel,
                title = title,
                legend = :none)
        end
        plot!(iters, trace;
            linealpha = linealpha,
            linecolor = linecolor,
            linestyle = linestyle,
            linewidth = linewidth,
            marker = marker,
            markersize = markersize,
            markershape = markershape,
            markeralpha = markeralpha,
            markercolor = markercolor,
            markerstrokealpha = markerstrokealpha,
            markerstrokecolor = markerstrokecolor,
            markerstrokestyle = markerstrokestyle,
            markerstrokewidth = markerstrokewidth)
    end;

    return plt
end

precompile(plot_trace, (Vector{FP}, Vector{FP}, ))
precompile(plot_trace, (Vector{FP}, Matrix{FP}, ))

function plot_trace_elbo(
    opt::OptSummary; 
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Evidence lower bound", 
    title::String = "Evidence lower bound evolution",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter))[burn:end]
    trace = opt.fitlog.elbo[burn:end]

    return plot_trace(
        iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end

function plot_trace_sigma2e(
    opt::OptSummary;
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Dispersion parameter", 
    title::String = "Dispersion parameter evolution",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter))[burn:end]
    trace = opt.fitlog.sigma2_e[burn:end]

    return plot_trace(iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end

function plot_trace_sigma2u(
    opt::OptSummary; 
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Random effect variance", 
    title::String = "Random effect variance evolution",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter))[burn:end]
    trace = opt.fitlog.sigma2_u[burn:end,:]

    return plot_trace(iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end

function plot_trace_regparam(
    opt::OptSummary; 
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Regression parameters", 
    title::String = "Regression parameter evolution",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter))[burn:end]
    trace = opt.fitlog.theta[burn:end,:]
    
    return plot_trace(iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end

function plot_trace_delta(
    opt::OptSummary; 
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Relative change", 
    title::String = "Relative change evolution",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter-1))[burn:end]
    trace = [opt.fitlog.dx opt.fitlog.df][burn:end,:]

    return plot_trace(iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end

function plot_trace_dx(
    opt::OptSummary; 
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Relative change", 
    title::String = "Parameter relative change",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter-1))[burn:end]
    trace = opt.fitlog.dx[burn:end]

    return plot_trace(iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end

function plot_trace_df(
    opt::OptSummary; 
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Relative change", 
    title::String = "Evidence lower bound relative change",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter-1))[burn:end]
    trace = opt.fitlog.df[burn:end]

    return plot_trace(iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end

function plot_trace_rate(
    opt::OptSummary; 
    burn::Int64 = 1,
    xlabel::String = "Iterations", 
    ylabel::String = "Relative change", 
    title::String = "Stepsize parameter evolution",
    add = false,
    kwargs...
    )

    iters = float.(collect(1:opt.niter-1))[burn:end]
    trace = opt.fitlog.rate[burn:end]

    return plot_trace(iters, trace;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        add = add,
        kwargs...)
end


precompile(plot_trace_elbo,     (OptSummary, ))
precompile(plot_trace_sigma2e,  (OptSummary, ))
precompile(plot_trace_sigma2u,  (OptSummary, ))
precompile(plot_trace_regparam, (OptSummary, ))
precompile(plot_trace_delta,    (OptSummary, ))
precompile(plot_trace_dx,       (OptSummary, ))
precompile(plot_trace_df,       (OptSummary, ))
precompile(plot_trace_rate,     (OptSummary, ))

