import math

def compute_central_angle(distance_km):
    # Earth's radius in kilometers
    radius_km = 6378.0

    # Circumference (just for info, not used in angle calc)
    circumference_km = 2 * math.pi * radius_km
    print("Earth's circumference = %.2f km" % circumference_km)

    # Angle in radians
    angle_radians = distance_km / radius_km

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# Example usage
if __name__ == "__main__":
    d = float(input("Enter distance between two points on Earth (in km): "))
    alpha = compute_central_angle(d)
    print("Central angle (α) = %.6f degrees" % alpha)

