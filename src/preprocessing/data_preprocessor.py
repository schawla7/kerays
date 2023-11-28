import pyarrow as pa
import ray
from ray.data.preprocessors import StandardScaler, OneHotEncoder, SimpleImputer, MinMaxScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self,target):
        self.target = target
        ray.init()

    def download_dataset(self, file_path):
        df = ray.data.read_csv(file_path)
        print("--------------- Before PreProcessing -----------------------")
        return df

    def get_feature_types(self, df):
        numerical_features = []
        categorical_features = []

        numerical_datatypes = [pa.int64(),pa.float64()]
        categorical_datatypes = [pa.string()]

        # Schema of Data
        schema = df.schema()
        field_types = schema.types
        field_names = schema.names

        for index,field in enumerate(field_names):
            if field!=self.target:
                dtype = field_types[index]
                if dtype in numerical_datatypes:
                    numerical_features.append(field)
                elif dtype in categorical_datatypes:
                    categorical_features.append(field)

        return numerical_features, categorical_features

    def preprocess_numerical_features(self, df, numerical_features):
        scaler = StandardScaler(columns=numerical_features)
        dataset_transformed = scaler.fit_transform(df)

        imputer = SimpleImputer(columns=numerical_features, strategy="constant",fill_value=0)
        dataset_transformed = imputer.fit_transform(dataset_transformed)
        return dataset_transformed

    def preprocess_categorical_features(self, df, categorical_features):
        encoder = OneHotEncoder(columns=categorical_features)
        dataset_transformed = encoder.fit_transform(df)

        target_encoder = OrdinalEncoder(columns=[self.target])
        dataset_transformed = target_encoder.fit_transform(dataset_transformed)
        return dataset_transformed

    def split_dataset(self,df):
        df = df.to_pandas()
        # Split dataset into train and test
        # Separating features and target variable
        # Sort the DataFrame if needed (for instance, by index)
        first_column_name = df.columns[0]
        df = df.sort_values(first_column_name)  # Replace 'some_column' with the column used for sorting if needed

        # Define 'target_column'
        target_column = self.target

        # Separate features (X) and target variable (y)
        X = df.drop(target_column, axis=1)  # Features
        y = df[target_column]  # Target variable

        # Calculate the index to split the data
        split_index = int(0.70 * len(df))  # 65% for training, 35% for testing

        # Splitting the data
        X_train = X.iloc[:split_index]  # First 65% for training features
        y_train = y.iloc[:split_index]  # First 65% for training target variable

        X_test = X.iloc[split_index:]  # Remaining 35% for testing features
        y_test = y.iloc[split_index:]  # Remaining 35% for testing target variable
        return X_train, y_train, X_test, y_test

    def preprocess_data(self, file_path):
        df = self.download_dataset(file_path)
        numerical_features, categorical_features = self.get_feature_types(df)
        df = self.preprocess_numerical_features(df, numerical_features)
        df = self.preprocess_categorical_features(df, categorical_features)

        X_train, y_train, X_test, y_test = self.split_dataset(df)

        return X_train, y_train, X_test, y_test
