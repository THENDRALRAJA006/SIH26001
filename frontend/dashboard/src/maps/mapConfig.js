/**
 * LAND-JEPA GIS Tile Configuration
 * Supports MapTiler high-resolution satellite, topographic contours, and dark GIS base layers
 * with graceful CARTO fallback.
 */

export const getMapTileLayer = (mode = "dark") => {
  const key = import.meta.env.VITE_MAPTILER_API_KEY || "";

  if (key) {
    if (mode === "satellite") {
      return {
        url: `https://api.maptiler.com/maps/hybrid/{z}/{x}/{y}.jpg?key=${key}`,
        attribution: '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 20,
      };
    }
    if (mode === "topo") {
      return {
        url: `https://api.maptiler.com/maps/topo-v2/{z}/{x}/{y}.png?key=${key}`,
        attribution: '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 20,
      };
    }
    if (mode === "light") {
      return {
        url: `https://api.maptiler.com/maps/streets-v2/{z}/{x}/{y}.png?key=${key}`,
        attribution: '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 20,
      };
    }
    // Default dark GIS dataviz
    return {
      url: `https://api.maptiler.com/maps/dataviz-dark/{z}/{x}/{y}.png?key=${key}`,
      attribution: '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 20,
    };
  }

  // Fallback to CARTO when no key is set
  if (mode === "light") {
    return {
      url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    };
  }
  return {
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    maxZoom: 19,
  };
};
