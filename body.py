from data import rows, fields
import numpy as np # Import numpy for trigonometric functions

class Planet:
    COLUMN_MAP = fields
    def __init__(self, position=(0,0,0), name="", row=[]):
        self.data = {}

        # --- Default/Placeholder Attributes ---
        self.name = name

        # Default to the 'position' arg or (0,0,0)
        self.position_cartesian = position

        # This dict holds the original astronomical coordinates
        self.position_celestial = {
            'ra_deg': None,
            'dec_deg': None,
            'distance_pc': None
        }

        # --- Populate from Data Row ---
        if len(row) == len(self.COLUMN_MAP):
            # Use zip to create a clean dictionary of all data
            self.data = dict(zip(self.COLUMN_MAP, row))

            # --- Function to safely convert data to float/None ---
            def safe_float(key):
                """Tries to convert a dictionary value to float, returns None on failure."""
                value = self.data.get(key)
                if value is None or value == '':
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None

            # --- Key Planet Attributes (using safe_float) ---
            self.name = self.data.get('pl_name')
            self.orbital_period_days = safe_float('pl_orbper')
            self.semi_major_axis_au = safe_float('pl_orbsmax')
            self.radius_earth = safe_float('pl_rade')
            self.mass_earth = safe_float('pl_bmasse')
            self.eccentricity = safe_float('pl_orbeccen')
            self.insolation_flux = safe_float('pl_insol') # Relative to Earth
            self.equilibrium_temp_k = safe_float('pl_eqt')

            # --- Key System & Host Star Attributes (using safe_float) ---
            self.host_name = self.data.get('hostname')
            # Use safe_float and try to convert to int later if needed, but float is safer
            self.num_stars_in_system = safe_float('sy_snum')
            self.num_planets_in_system = safe_float('sy_pnum')
            self.host_spectral_type = self.data.get('st_spectype')
            self.host_temp_k = safe_float('st_teff')
            self.host_radius_solar = safe_float('st_rad')
            self.host_mass_solar = safe_float('st_mass')

            # --- Position & Discovery Attributes ---
            self.position_celestial['ra_deg'] = safe_float('ra')
            self.position_celestial['dec_deg'] = safe_float('dec')
            self.position_celestial['distance_pc'] = safe_float('sy_dist')

            self.discovery_method = self.data.get('discoverymethod')
            self.discovery_year = self.data.get('disc_year')
            self.discovery_facility = self.data.get('disc_facility')

            # --- Calculate 3D Cartesian Coordinates (Protected) ---
            try:
                # Get the safely converted data
                ra = self.position_celestial['ra_deg']
                dec = self.position_celestial['dec_deg']
                distance = self.position_celestial['distance_pc']

                # Ensure all necessary values are present (not None)
                if None not in (ra, dec, distance):
                    # Convert RA and Dec from degrees to radians
                    ra_rad = np.radians(ra)
                    dec_rad = np.radians(dec)

                    # Convert to 3D Cartesian coordinates (X, Y, Z)
                    X = distance * np.cos(dec_rad) * np.cos(ra_rad)
                    Y = distance * np.cos(dec_rad) * np.sin(ra_rad)
                    Z = distance * np.sin(dec_rad)

                    # Overwrite the default position with the calculated one
                    self.position_cartesian = (X, Y, Z)

            except Exception:
                # Catches any remaining errors (e.g., if numpy isn't available)
                # self.position_cartesian keeps its default (0,0,0).
                pass

        elif len(row) > 0:
            # Handle error if a row is provided but its length is wrong
            print(f"Error: Row length ({len(row)}) does not match expected column length ({len(self.COLUMN_MAP)}).")
            print(f"Planet '{name}' not fully initialized.")

    def __repr__(self):
        """Provides a clean string representation of the planet object."""
        if self.name and self.host_name:
            return f"<Planet: {self.name} (Host: {self.host_name})>"
        elif self.name:
            return f"<Planet: {self.name}>"
        else:
            return "<Planet: Uninitialized>"

planets = [Planet(row=i) for i in rows]
