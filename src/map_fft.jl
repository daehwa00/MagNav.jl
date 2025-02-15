"""
    upward_fft(map_map::Matrix, dx, dy, dz; expand::Bool=true, α=0)

Upward continuation of a potential field (i.e., magnetic anomaly field) map.
Uses the Fast Fourier Transform (FFT) to convert the map to the frequency
domain, applies an upward continuation filter, and uses the inverse FFT to
convert the map back to the spatial domain. Optionally expands the map
temporarily with periodic padding. Downward continuation may be performed to a
limited degree as well, but be careful, as this is generally unstable and
amplify high frequencies (i.e., noise).

Reference: Blakely, Potential Theory in Gravity and Magnetic Applications,
2009, Chapter 12 & Appendix B (pg. 315-317 & 402).

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `dx:`:     x-direction map step size [m]
- `dy`:      y-direction map step size [m]
- `dz`:      z-direction upward/downward continuation distance(s) [m]
- `expand`:  (optional) if true, expand map temporarily to reduce edge effects
- `α`:       (optional) regularization parameter for downward continuation

**Returns:**
- `map_map`: `ny` x `nx` 2D gridded map data, upward/downward continued
"""
function upward_fft(map_map::Matrix, dx, dy, dz; expand::Bool=true, α=0)

    (ny,nx) = size(map_map)

    if expand
        pad = min(maximum(ceil.(Int,10*maximum(abs.(dz))./(dx,dy))),5000) # set pad > 10*dz
        (map_,px,py) = map_expand(map_map,pad) # expand with pad
        (Ny,Nx) = size(map_)
    else
        map_ = map_map
        (Ny,Nx,px,py) = (ny,nx,0,0)
    end

    map_map = length(dz) == 1 ? deepcopy(map_map) : repeat(map_map,1,1,length(dz))
    all(dz .≈ 0) && return map_map

    (k,_,_) = create_k(dx,dy,Nx,Ny) # radial wavenumber grid
    map_    = fft(map_) # FFT
    for (i,dz_i) in enumerate(dz)
        α_i = dz_i > 0 ? 0 : α # ensure no regularization if upward
        H = exp.(-k.*dz_i) ./ (1 .+ α_i .* k.^2 .* exp.(-k.*dz_i))  # filter
        map_map[:,:,i] = real(ifft(map_.*H))[(1:ny).+py,(1:nx).+px] # inverse FFT
    end

    return (map_map)
end # function upward_fft

"""
    upward_fft(map_map::Map, alt; expand::Bool=true, α=0)

Upward continuation of a potential field (i.e., magnetic anomaly field) map.
Uses the Fast Fourier Transform (FFT) to convert the map to the frequency
domain, applies an upward continuation filter, and uses the inverse FFT to
convert the map back to the spatial domain. Optionally expands the map
temporarily with periodic padding. Downward continuation may be performed to a
limited degree as well, but be careful, as this is generally unstable and
amplify high frequencies (i.e., noise).

Reference: Blakely, Potential Theory in Gravity and Magnetic Applications,
2009, Chapter 12 & Appendix B (pg. 315-317 & 402).

**Arguments:**
- `map_map`: `Map` magnetic anomaly map struct
- `alt`:     target upward continuation altitude(s) [m]
- `expand`:  (optional) if true, expand map temporarily to reduce edge effects
- `α`:       (optional) regularization parameter for downward continuation

**Returns:**
- `map_map`: `Map` magnetic anomaly map struct, upward/downward continued (`MapS` with `alt` vector => `MapS3D`)
"""
function upward_fft(map_map::Map, alt; expand::Bool=true, α=0)

    if length(alt) > 1
        @assert typeof(map_map) <: Union{MapS,MapS3D} "multiple upward continuation altitudes only allowed for MapS or MapS3D"
        alt = sort(alt)
    end

    alt = convert.(eltype(map_map.alt),alt)
    dx  = dlon2de(get_step(map_map.xx),mean(map_map.yy))
    dy  = dlat2dn(get_step(map_map.yy),mean(map_map.yy))

    if (typeof(map_map) <: Union{MapS,MapSd,MapV}) & (all(alt .>= median(map_map.alt)) | (α > 0))

        dz = alt .- median(map_map.alt)

        if typeof(map_map) <: Union{MapS,MapSd} # scalar map
            if length(alt) > 1
                map_map = MapS3D(upward_fft(map_map.map,dx,dy,dz,expand=expand,α=α),
                                 map_map.xx,map_map.yy,alt)
            else # 3D map
                map_map = MapS(upward_fft(map_map.map,dx,dy,dz,expand=expand,α=α),
                               map_map.xx,map_map.yy,alt)
            end
        elseif typeof(map_map) <: MapV # vector map
            mapX    = upward_fft(map_map.mapX,dx,dy,dz,expand=expand,α=α)
            mapY    = upward_fft(map_map.mapY,dx,dy,dz,expand=expand,α=α)
            mapZ    = upward_fft(map_map.mapZ,dx,dy,dz,expand=expand,α=α)
            map_map = MapV(mapX,mapY,mapZ,map_map.xx,map_map.yy,alt)
        end

    elseif (typeof(map_map) <: Union{MapS3D}) & (all(alt .>= map_map.alt[1]) | (α > 0))

        (ny,nx,_) = size(map_map.map)
        alt  = [alt;] # ensure vector
        dalt = get_step(map_map.alt)
        @assert all(rem.(alt .- map_map.alt[1],dalt) .≈ 0) "alt must have same step size as in MapS3D alt"
        alt_down = alt[alt .< map_map.alt[1]]
        alt_up   = alt[alt .> map_map.alt[end]]

        if length(alt_down) > 0 # downward continue from lowest map
            dz       = alt_down .- map_map.alt[1]
            map_down = upward_fft(map_map.map[:,:,1],dx,dy,dz,expand=expand,α=α)
        else
            map_down = Array{eltype(alt)}(undef,ny,nx,0)
        end

        if length(alt_up) > 0 # upward continue from highest map
            dz     = alt_up .- map_map.alt[end]
            map_up = upward_fft(map_map.map[:,:,end],dx,dy,dz,expand=expand,α=α)
        else
            map_up = Array{eltype(alt)}(undef,ny,nx,0)
        end

        map_map = MapS3D(cat(map_down,map_map.map,map_up,dims=3),map_map.xx,map_map.yy,
                            [alt_down;map_map.alt;alt_up])

    else
        @info("α must be specified for downward continuation, returning original map")
    end

    return (map_map)
end # function upward_fft

"""
    vector_fft(map_map::Matrix, dx, dy, D, I)

Get potential field (i.e., magnetic anomaly field) map vector components
using declination and inclination.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `dx`:      x-direction map step size [m]
- `dy`:      y-direction map step size [m]
- `D`:       map declination (Earth core field) [rad]
- `I`:       map inclination (Earth core field) [rad]

**Returns:**
- `Bx, By, Bz`: map vector components
"""
function vector_fft(map_map::Matrix, dx, dy, D, I)
    (ny,nx) = size(map_map)
    (s,u,v) = create_k(dx,dy,nx,ny)

    l = cos(I)*cos(D)
    m = cos(I)*sin(D)
    n = sin(I)

    F = fft(map_map)

    Hx = im*u ./ (im*(u*l+m*v)+n*s)
    Hy = im*v ./ (im*(u*l+m*v)+n*s)
    Hz = s    ./ (im*(u*l+m*v)+n*s)

    Hx[1,1] = 1
    Hy[1,1] = 1
    Hz[1,1] = 1

    Bx = real(ifft(Hx.*F))
    By = real(ifft(Hy.*F))
    Bz = real(ifft(Hz.*F))

    return (Bx, By, Bz)
end # function vector_fft

"""
    create_k(dx, dy, nx::Int, ny::Int)

Internal helper function to create radial wavenumber (spatial frequency) grid.

**Arguments:**
- `dx`: x-direction map step size [m]
- `dy`: y-direction map step size [m]
- `nx`: x-direction map dimension [-]
- `ny`: y-direction map dimension [-]

**Returns:**
- `k`:  `ny` x `nx` radial wavenumber (i.e., magnitude of wave vector)
- `kx`: `ny` x `nx` x-direction radial wavenumber
- `ky`: `ny` x `nx` y-direction radial wavenumber
"""
function create_k(dx, dy, nx::Int, ny::Int)
    # DFT sample frequencies [rad/m], 1/dx & 1/dy are sampling rates [1/m]
    kx = nx*dx==0 ? zeros(ny,nx) : repeat(2*pi*fftfreq(nx,1/dx)',ny,1)
    ky = ny*dy==0 ? zeros(ny,nx) : repeat(2*pi*fftfreq(ny,1/dy) ,1,nx)
    k  = sqrt.(kx.^2+ky.^2)
    return (k, kx, ky)
end # function create_k

"""
    map_expand(map_map::Matrix, pad::Int=1)

Internal helper function to expand a map with padding on each edge to eliminate
discontinuities in the discrete Fourier transform. The map is “wrapped around”
to make it periodic. Padding expands the map to 7-smooth dimensions, allowing
for a faster Fast Fourier Transform algorithm to be used during upward/downward
continuation.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `pad`:     minimum padding (grid cells) along map edges

**Returns:**
- `map_map`: `ny` x `nx` 2D gridded map data, expanded (padded)
- `padx`:    x-direction padding (grid cells) applied on first edge
- `pady`:    y-direction padding (grid cells) applied on first edge
"""
function map_expand(map_map::Matrix, pad::Int=1)

    map_ = deepcopy(map_map)

    (ny,nx) = size(map_) # original map size
    (Ny,Nx) = smooth7.((ny,nx).+2*pad) # map size with 7-smooth padding
    # (Ny,Nx) = (ny,nx).+ 2*pad # map size with naive padding

    # padding on each edge
    padx = (floor(Int,(Nx-nx)/2),ceil(Int,(Nx-nx)/2))
    pady = (floor(Int,(Ny-ny)/2),ceil(Int,(Ny-ny)/2))

    # place original map in middle of new map
    (x1,x2) = (1,nx) .+ padx[1]
    (y1,y2) = (1,ny) .+ pady[1]
    map_map = zeros(Ny,Nx)
    map_map[y1:y2,x1:x2] = map_

    # fill row edges (right/left)
    for i = y1:y2
        vals = LinRange(map_map[i,x1],map_map[i,x2],Nx-nx+2)[2:end-1]
        map_map[i,1:x1-1  ] = reverse(vals[1:1:padx[1]])
        map_map[i,x2+1:end] = reverse(vals[(1:padx[2]).+padx[1]])
    end

    # fill column edges (top/bottom)
    for i = 1:Nx
        vals = LinRange(map_map[y1,i],map_map[y2,i],Ny-ny+2)[2:end-1]
        map_map[1:y1-1  ,i] = reverse(vals[1:1:pady[1]])
        map_map[y2+1:end,i] = reverse(vals[(1:pady[2]).+pady[1]])
    end

    return (map_map, padx[1], pady[1])
end # function map_expand

"""
    smooth7(x::Int)

Internal helper function to find the lowest 7-smooth number `y` >= `x`.
"""
function smooth7(x::Int)
    y = 2*x
    for i = 0:ceil(Int,log(7,x))
        for j = 0:ceil(Int,log(5,x))
            for k = 0:ceil(Int,log(3,x))
                z = 7^i*5^j*3^k
                z < 2*x && (y = min(y, 2^ceil(Int,log(2,x/z))*z))
            end
        end
    end
    return (y)
end # function smooth7

"""
    downward_L(map_map::Matrix, dx, dy, dz, α::Vector;
               expand::Bool = true,
               ind          = map_params(map_map)[2])

Downward continuation using a sequence of regularization parameters to create
a characteristic L-curve. The optimal regularization parameter is at a local
minimum on the L-curve, which is a local maximum of curvature. The global
maximum of curvature may or may not be the optimal regularization parameter.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `dx:`:     x-direction map step size [m]
- `dy`:      y-direction map step size [m]
- `dz`:      z-direction upward/downward continuation distance [m]
- `α`:      (geometric) sequence of regularization parameters
- `expand`: (optional) if true, expand map temporarily to reduce edge effects
- `ind`:    (optional) selected map indices (e.g., non-missing data)

**Returns:**
- `norms`: L-infinity norm of difference between sequential D.C. solutions
"""
function downward_L(map_map::Matrix, dx, dy, dz, α::Vector;
                    expand::Bool = true,
                    ind          = map_params(map_map)[2])

    (ny,nx) = size(map_map)
    norms   = zeros(length(α)-1)

    if expand
        pad = min(maximum(ceil.(Int,10*abs(dz)./(dx,dy))),5000) # set pad > 10*dz
        (map_map,px,py) = map_expand(map_map,pad)     # expand with pad
        (Ny,Nx) = size(map_map)
    else
        (Ny,Nx,px,py) = (ny,nx,0,0)
    end

    (k,_,_) = create_k(dx,dy,Nx,Ny) # radial wavenumber grid
    H_temp  = exp.(-k.*dz)
    H       = H_temp ./ (1 .+ α[1] .* k.^2 .* H_temp) # filter
    map_old = real(ifft(fft(map_map).*H))
    map_old = map_old[(1:ny).+py,(1:nx).+px][ind]
    for i = 2:length(α)
        H       = H_temp ./ (1 .+ α[i] .* k.^2 .* H_temp) # filter
        map_new = real(ifft(fft(map_map).*H))
        map_new = map_new[(1:ny).+py,(1:nx).+px][ind]
        norms[i-1] = norm(map_new-map_old,Inf)
        map_old = map_new
    end

    return (norms)
end # function downward_L

"""
    downward_L(mapS::Union{MapS,MapSd,MapS3D}, alt, α::Vector;
               expand::Bool = true,
               ind          = map_params(mapS)[2])

Downward continuation using a sequence of regularization parameters to create
a characteristic L-curve. The optimal regularization parameter is at a local
minimum on the L-curve, which is a local maximum of curvature. The global
maximum of curvature may or may not be the optimal regularization parameter.

**Arguments:**
- `mapS`:   `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct
- `alt`:    target downward continuation altitude [m]
- `α`:      (geometric) sequence of regularization parameters
- `expand`: (optional) if true, expand map temporarily to reduce edge effects
- `ind`:    (optional) selected map indices (e.g., non-missing data)

**Returns:**
- `norms`: L-infinity norm of difference between sequential D.C. solutions
"""
function downward_L(mapS::Union{MapS,MapSd,MapS3D}, alt, α::Vector;
                    expand::Bool = true,
                    ind          = map_params(mapS)[2])
    dx   = dlon2de(get_step(mapS.xx),mean(mapS.yy))
    dy   = dlat2dn(get_step(mapS.yy),mean(mapS.yy))
    alt_ = typeof(mapS) <: Union{MapSd} ? median(mapS.alt[ind]) : mapS.alt[1]
    dz   = alt - alt_
    typeof(mapS) <: MapS3D && @info("3D map provided, using map at lowest altitude")
    return downward_L(mapS.map[:,:,1],dx,dy,dz,α;
                      expand = expand,
                      ind    = ind)
end # function downward_L

"""
    psd(map_map::Matrix, dx, dy)

Power spectral density of a potential field (i.e., magnetic anomaly field) map.
Uses the Fast Fourier Transform to determine the spectral energy distribution
across the radial wavenumbers (spatial frequencies) in the Fourier transform.

**Arguments:**
- `map_map`: `ny` x `nx` 2D gridded map data
- `dx`:      x-direction map step size [m]
- `dy`:      y-direction map step size [m]

**Returns:**
- `map_psd`: `ny` x `nx` power spectral density of 2D gridded map data
- `kx`:      `ny` x `nx` x-direction radial wavenumber
- `ky`:      `ny` x `nx` y-direction radial wavenumber
"""
function psd(map_map::Matrix, dx, dy)
    (ny,nx)   = size(map_map)
    (_,kx,ky) = create_k(dx,dy,nx,ny)
    map_psd   = abs.(fft(map_map)).^2
    return (map_psd, kx, ky)
end # function psd

"""
    psd(mapS::Union{MapS,MapSd,MapS3D})

Power spectral density of a potential field (i.e., magnetic anomaly field) map.
Uses the Fast Fourier Transform to determine the spectral energy distribution
across the radial wavenumbers (spatial frequencies) in the Fourier transform.

**Arguments:**
- `mapS`: `MapS`, `MapSd`, or `MapS3D` scalar magnetic anomaly map struct

**Returns:**
- `map_psd`: `ny` x `nx` power spectral density of 2D gridded map data
- `kx`:      `ny` x `nx` x-direction radial wavenumber
- `ky`:      `ny` x `nx` y-direction radial wavenumber
"""
function psd(mapS::Union{MapS,MapSd,MapS3D})
    dx = dlon2de(get_step(mapS.xx),mean(mapS.yy))
    dy = dlat2dn(get_step(mapS.yy),mean(mapS.yy))
    typeof(mapS) <: MapS3D && @info("3D map provided, using map at lowest altitude")
    return psd(mapS.map[:,:,1], dx, dy)
end # function psd
