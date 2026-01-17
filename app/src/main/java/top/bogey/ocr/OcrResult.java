package top.bogey.ocr;

import android.graphics.Rect;
import android.os.Parcel;
import android.os.Parcelable;

public class OcrResult implements Parcelable {
    private final Rect area;
    private final String text;
    private final int similar;

    public OcrResult(Rect area, String text, int similar) {
        this.area = area;
        this.text = text;
        this.similar = similar;
    }

    protected OcrResult(Parcel in) {
        area = in.readParcelable(Rect.class.getClassLoader());
        text = in.readString();
        similar = in.readInt();
    }

    public static final Creator<OcrResult> CREATOR = new Creator<>() {
        @Override
        public OcrResult createFromParcel(Parcel in) {
            return new OcrResult(in);
        }

        @Override
        public OcrResult[] newArray(int size) {
            return new OcrResult[size];
        }
    };

    public Rect getArea() {
        return area;
    }

    public int getSimilar() {
        return similar;
    }

    public String getText() {
        return text;
    }

    @Override
    public String toString() {
        return "OcrResult{" +
                "area=" + area +
                ", text='" + text + '\'' +
                ", similar=" + similar +
                '}';
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeParcelable(area, flags);
        dest.writeString(text);
        dest.writeInt(similar);
    }
}
