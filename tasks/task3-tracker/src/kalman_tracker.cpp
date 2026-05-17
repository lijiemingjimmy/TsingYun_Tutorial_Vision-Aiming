#include "kalman_tracker.hpp"

#include <stdexcept>
#include <algorithm>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;

    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

    void KalmanTracker::AxisFilter::predict(double dt, double process_noise)
    {
        dt = std::max(dt, 0.0);
        position += velocity * dt;

        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;

        double q00 = process_noise * dt4 / 4.0;
        double q01 = process_noise * dt3 / 2.0;
        double q10 = process_noise * dt3 / 2.0;
        double q11 = process_noise * dt2;

        double new_p00 = p00 + dt * p10 + dt * (p01 + dt * p11) + q00;
        double new_p01 = p01 + dt * p11 + q01;
        double new_p10 = p10 + dt * p11 + q10;
        double new_p11 = p11 + q11;

        p00 = new_p00;
        p01 = new_p01;
        p10 = new_p10;
        p11 = new_p11;
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise)
    {
        double residual = measured_position - position;
        double S = p00 + measurement_noise;

        if (S <= 0.0)
        {
            return;
        }

        double K0 = p00 / S;
        double K1 = p10 / S;

        position += K0 * residual;
        velocity += K1 * residual;

        double new_p00 = (1.0 - K0) * p00;
        double new_p01 = (1.0 - K0) * p01;
        double new_p10 = -K1 * p00 + p10;
        double new_p11 = -K1 * p01 + p11;

        p00 = new_p00;
        p01 = new_p01;
        p10 = new_p10;
        p11 = new_p11;
    }

    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
        if (!tracking_)
        {
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);
            tracking_ = true;
            return stateFromFilters();
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);

        return stateFromFilters();
    }

    TrackState KalmanTracker::predict(double dt)
    {
        if (!tracking_)
        {
            return TrackState{false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        return stateFromFilters();
    }

    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw
