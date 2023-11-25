import pyarrow as pa
import ray
from ray.data.preprocessors import StandardScaler, OneHotEncoder, SimpleImputer, MinMaxScaler


class DataPreprocessor:
    def __init__(self,target):
        self.target = target
        ray.init(num_cpus=4)

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

        imputer = SimpleImputer(columns=numerical_features, strategy="mean")
        dataset_transformed = imputer.fit_transform(dataset_transformed)

        return dataset_transformed

    def preprocess_categorical_features(self, df, categorical_features):
        encoder = OneHotEncoder(columns=categorical_features)
        dataset_transformed = encoder.fit_transform(df)
        return dataset_transformed

    def split_dataset(self,df):
        # Split dataset into train and test
        train_df, test_df = df.train_test_split(test_size=0.35,seed=420)

        train_df = train_df.to_pandas()
        test_df = test_df.to_pandas()

        # Obtain X and Y splits
        X_train, y_train = train_df.drop(self.target, axis=1), train_df[self.target]
        X_test, y_test = test_df.drop(self.target, axis=1), test_df[self.target]
        return X_train, y_train, X_test, y_test

    def preprocess_data(self, file_path):
        df = self.download_dataset(file_path)
        numerical_features, categorical_features = self.get_feature_types(df)
        df = self.preprocess_numerical_features(df, numerical_features)
        df = self.preprocess_categorical_features(df, categorical_features)

        X_train, y_train, X_test, y_test = self.split_dataset(df)

        return X_train, y_train, X_test, y_test
