
module ENF_TEST
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/examples/enf_2d.jl"))
        end
    end
end
