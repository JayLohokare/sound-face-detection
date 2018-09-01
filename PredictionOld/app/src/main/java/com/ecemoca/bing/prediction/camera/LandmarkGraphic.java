package com.ecemoca.bing.prediction.camera;

/**
 * Created by bing on 12/31/17.
 */

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;

/**
 * Graphics class for rendering Googly Eyes on a graphic overlay given the current eye positions.
 */
class LandmarkGraphic extends GraphicOverlay.Graphic {
    private Paint mLandmarkPaint;
    private Paint mEyeRedPaint;
    private Paint mFacePaint;
    private Paint mFaceReference;
    private Paint mIdPaint;
    private volatile PointF mPosition;
    private volatile PointF[] mLandmarks = null;
    private volatile float mHeight, mWidth;
    private boolean alignment = false;
    private float[] landmarks = null;

    //==============================================================================================
    // Methods
    //==============================================================================================

    LandmarkGraphic(GraphicOverlay overlay) {
        super(overlay);

        mLandmarkPaint = new Paint();
        mLandmarkPaint.setColor(Color.YELLOW);
        mLandmarkPaint.setStyle(Paint.Style.FILL);

        mEyeRedPaint = new Paint();
        mEyeRedPaint.setColor(Color.RED);
        mEyeRedPaint.setStyle(Paint.Style.FILL);

        mFacePaint = new Paint();
        mFacePaint.setColor(Color.RED);
        mFacePaint.setStyle(Paint.Style.STROKE);
        mFacePaint.setStrokeWidth(5);

        mFaceReference = new Paint();
        mFaceReference.setColor(Color.GREEN);
        mFaceReference.setStyle(Paint.Style.STROKE);
        mFaceReference.setStrokeWidth(5);

        mIdPaint = new Paint();
        mIdPaint.setColor(Color.GRAY);
        mIdPaint.setTextSize(50);

    }

    /**
     * Updates the eye positions and state from the detection of the most recent frame.  Invalidates
     * the relevant portions of the overlay to trigger a redraw.
     */
    void updateEyes(PointF[] landmarks,
                    PointF position, float height, float width) {
        mLandmarks = landmarks;
        mPosition = position;
        mHeight = height;
        mWidth = width;

        postInvalidate();
    }

    /**
     * Draws the current eye state to the supplied canvas.  This will draw the eyes at the last
     * reported position from the tracker, and the iris positions according to the physics
     * simulations for each iris given motion and other forces.
     */
    @Override
    public void draw(Canvas canvas) {
        PointF detectPosition = mPosition;
        float detectWidth = mWidth;
        float detectHeight = mHeight;

//        drawLandmarkReference(canvas);
        drawFaceRegion(canvas, detectPosition, detectHeight, detectWidth);

        if (mLandmarks == null) {
            return;
        }

        int i = 0;
        landmarks = new float[16];
        for (PointF point : mLandmarks) {
            if (point == null) {
                i++; i++;
                return;
            }
            PointF position = new PointF(translateX(point.x), translateY(point.y));
            canvas.drawCircle(position.x, position.y, 10, mLandmarkPaint);
            landmarks[i++] = position.x;
            landmarks[i++] = position.y;
        }
    }

    private void drawLandmarkReference(Canvas canvas) {
        int mRadius = 7;
        float offsetx = 50f;
        float offsety = 50f;

        canvas.drawCircle(562 + offsetx, 586.7f + offsety, 2*mRadius, mEyeRedPaint);
        canvas.drawCircle(361f + offsetx, 595f + offsety, 2*mRadius, mEyeRedPaint);
        canvas.drawCircle(584.5f + offsetx, 708f + offsety, 2*mRadius, mEyeRedPaint);
        canvas.drawCircle(458.6f + offsetx, 738.56f + offsety, 2*mRadius, mEyeRedPaint);
        canvas.drawCircle(347.4f + offsetx, 700.25f + offsety, 2*mRadius, mEyeRedPaint);
        canvas.drawCircle(552.8f + offsetx, 837.8f + offsety, 2*mRadius, mEyeRedPaint);
        canvas.drawCircle(465f + offsetx, 889.7f + offsety,2*mRadius, mEyeRedPaint);
        canvas.drawCircle(379.9f + offsetx, 841.27f + offsety, 2*mRadius, mEyeRedPaint);
    }

    private void drawFaceRegion(Canvas canvas, PointF position, float height, float width) {
        float x = translateX(position.x + width / 2);
        float y = translateY(position.y + height / 2);
        // Draws a bounding box around the face.
        float xOffset = scaleX(width / 2.0f);
        float yOffset = scaleY(height / 2.0f);
        float left = x - xOffset;
        float top = y - yOffset;
        float right = x + xOffset;
        float bottom = y + 1.2f*yOffset;
        float xthred = 280, ythred = 380;
        // Draws a bounding box around the face.
        float xr = 540, yr = 650;
        canvas.drawRect(xr - xthred, yr - yOffset/xOffset*xthred, xr + xthred, yr + yOffset/xOffset*xthred*1.2f, mFaceReference);
        canvas.drawRect(xr - ythred, yr - yOffset/xOffset*ythred, xr + ythred, yr + yOffset/xOffset*ythred*1.2f, mFaceReference);
        canvas.drawRect(left, top, right, bottom, mFacePaint);

        // Check alignment status
        if (left < xr - xthred && left > xr - ythred && right > xr + xthred && right < xr + ythred
                && top > yr - yOffset/xOffset*ythred && top < yr - yOffset/xOffset*xthred) {
            canvas.drawText("Face Aligned", 400, 50, mIdPaint);
            alignment = true;
        }
        else {
            alignment = false;
        }
    }

    public float[] getLandmarks() {
        if (alignment)
            return landmarks;
        else
            return new float[16];
    }
}
