export IGAShellWeakBC
#
#
#

struct IGAShellWeakBC{P<:IGAShell} <: Five.AbstractExternalForce
    faceset::GeometryObjectVectors
    prescribed_displacement::Function
    components::Vector{Int}
    igashell::Base.RefValue{P}
    penalty::Float64
end

function IGAShellWeakBC(; 
    set, 
    func::Function,
    comps::AbstractVector{Int}, 
    igashell::IGAShell{dim_p,dim_s,T},
    penalty::T
    ) where {dim_p,dim_s,T}
    
    @assert( length(comps) == length(func(zero(Vec{dim_s,T}), 0.0)) )

    return IGAShellWeakBC{typeof(igashell)}(collect(set), func, collect(comps), Base.RefValue(igashell), penalty)
end

function Five.init_external_force!(a::IGAShellWeakBC, ::Ferrite.AbstractDofHandler)
    return a
end

function Five.apply_external_force!(wb::IGAShellWeakBC, state::StateVariables{T}, globaldata) where {T}
    
    
    #Igashell extract
    dh = globaldata.dh
    igashell = wb.igashell[]
    dim_s = Ferrite.getdim(igashell)

    #Coords
    ncoords = Ferrite.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, ncoords)
    Xᵇ = zeros(Vec{dim_s,T}, ncoords)
    
    A = 0.0
    for (ic, faceidx) in enumerate(wb.faceset)
        cellid, faceid = faceidx

        ndofs = ndofs_per_cell(dh, cellid)
        
        celldofs = zeros(Int, ndofs)
        fe = zeros(T, ndofs)
        ke = zeros(T, ndofs, ndofs)

        local_cellid = findfirst((i)->i==cellid, igashell.cellset)
        #
        Ce = get_extraction_operator(intdata(igashell), local_cellid)
        
        #Get coords and dofs of cell
        Ferrite.cellcoords!(X, dh, cellid)
        Ferrite.celldofs!(celldofs, dh, cellid)
        Xᵇ .= IGA.compute_bezier_points(Ce, X)

        ue = state.d[celldofs]
        
        #
        @timeit "build" cv = build_facevalue!(igashell, faceidx)

        IGA.set_bezier_operator!(cv, Ce)
        reinit!(cv, Xᵇ)

        @timeit "integrate" A += _compute_igashell_weak_boundary_condition!(cv, Xᵇ, wb.prescribed_displacement, wb.components, faceidx, state.t, ke, fe, ue, wb.penalty, getwidth(layerdata(igashell)))
        
        state.system_arrays.fᵉ[celldofs] += fe
        state.system_arrays.Kᵉ[celldofs, celldofs] += ke
        #@show celldofs[(fe .≈ 0.0).==false]
    end

end

function _compute_igashell_weak_boundary_condition!(fv::IGAShellValues, coords::AbstractVector{Vec{dim,T}}, prescr_disp::Function, components::Vector{Int}, faceidx, time::T, ke::AbstractMatrix, fe::AbstractVector, ue::AbstractVector, stiffness::T, width::T) where {dim,T}
    
    dA = 0.0
    for q_point in 1:getnquadpoints(fv)
        dΓ = getdetJdA(fv, q_point, faceidx) * width
        u = function_value(fv, q_point, ue)
        
        X = spatial_coordinate(fv, q_point, coords)
        g = prescr_disp(X,time)
        #dΓ = norm(cross(fv.G[q_point][2], fv.G[q_point][3])) * fv.qr.weights[q_point]
        dA += dΓ
        for i in 1:dim:getnbasefunctions(fv)
            δui = basis_value(fv, q_point, i)  
            
            for (c,d1) in enumerate(components)
                ug = u[d1] - g[c]
                fe[i+d1-1] += stiffness*ug*δui[1] * dΓ
            end

            for j in 1:dim:getnbasefunctions(fv)   
                δuj = basis_value(fv, q_point, j)

                for d1 in components
                    ke[i+d1-1,j+d1-1] += stiffness*δui[1]*δuj[1] * dΓ
                end
            end
        end
    end
    #@show norm(ke)
    #@show dA
    return dA

end
